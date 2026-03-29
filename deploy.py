import html
import json
from pathlib import Path
import textwrap

import cv2
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

from model_utils import (
    LEGACY_IMG_SIZE,
    LEGACY_MODEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    build_classifier_model,
    get_model_image_size,
)


st.set_page_config(
    page_title="Nebula Lens | AI Image Classifier",
    layout="wide",
)


DEFAULT_DECISION_THRESHOLD = 0.5
DECISION_MARGIN = 0.08
PORTRAIT_DECISION_THRESHOLD = 0.7


def load_training_summary():
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return {}


TRAINING_SUMMARY = load_training_summary()
SAVED_DECISION_THRESHOLD = float(
    TRAINING_SUMMARY.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
)
DECISION_THRESHOLD = max(DEFAULT_DECISION_THRESHOLD, SAVED_DECISION_THRESHOLD)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        model = build_classifier_model()
        try:
            model.load_weights(MODEL_PATH)
            return model
        except Exception:
            st.warning("Could not load AIGeneratedModel.weights.h5. Trying legacy model file.")

    if LEGACY_MODEL_PATH.exists():
        try:
            return keras.models.load_model(LEGACY_MODEL_PATH, compile=False)
        except Exception:
            try:
                legacy_model = build_classifier_model(img_size=LEGACY_IMG_SIZE)
                legacy_model.load_weights(LEGACY_MODEL_PATH)
                return legacy_model
            except Exception:
                st.warning("Could not load AIGeneratedModel.h5. Using an untrained model instead.")
    else:
        st.warning("No trained .h5 model file was found. Using an untrained model instead.")

    return build_classifier_model()


@st.cache_resource
def load_content_model():
    with st.spinner('Loading advanced image recognition model...'):
        return EfficientNetB0(weights='imagenet')


@st.cache_resource
def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    return detector


def detect_face_count(image):
    try:
        detector = load_face_detector()
        if detector is None:
            return 0

        rgb_image = np.array(image.convert("RGB"))
        grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if faces is None:
            return 0
        return int(len(faces))
    except Exception:
        return 0


def get_effective_threshold(face_count):
    threshold = DECISION_THRESHOLD
    if face_count > 0:
        threshold = max(threshold, PORTRAIT_DECISION_THRESHOLD)
    return float(threshold)


def get_decision_bounds(face_count):
    real_floor = float(DECISION_THRESHOLD)
    ai_cutoff = float(get_effective_threshold(face_count))
    return {
        "real_floor": real_floor,
        "ai_cutoff": ai_cutoff,
    }


def get_final_decision(confidence, bounds, face_count):
    if face_count > 0 and bounds["ai_cutoff"] > bounds["real_floor"]:
        if confidence >= bounds["ai_cutoff"]:
            return "ai_generated"
        if confidence <= bounds["real_floor"]:
            return "real"
        return "uncertain"
    if confidence >= bounds["ai_cutoff"] + DECISION_MARGIN:
        return "ai_generated"
    if confidence <= bounds["real_floor"] - DECISION_MARGIN:
        return "real"
    return "uncertain"


def get_threshold_gap_text(confidence, bounds, face_count, final_decision):
    real_floor = bounds["real_floor"]
    ai_cutoff = bounds["ai_cutoff"]

    if final_decision == "ai_generated":
        return f"{format_points(confidence - ai_cutoff)} above AI cutoff"
    if final_decision == "real":
        return f"{format_points(real_floor - confidence)} below real floor"
    if face_count > 0 and ai_cutoff > real_floor:
        return f"between {format_percent(real_floor)} and {format_percent(ai_cutoff)}"

    distance = abs(confidence - ai_cutoff)
    if distance == 0:
        return "exactly on the cutoff"
    direction = "below cutoff" if confidence < ai_cutoff else "above cutoff"
    return f"{format_points(distance)} {direction}"


def get_review_lean(confidence, bounds, face_count):
    if not (face_count > 0 and bounds["ai_cutoff"] > bounds["real_floor"]):
        return None

    midpoint = (bounds["real_floor"] + bounds["ai_cutoff"]) / 2
    if confidence > midpoint:
        return "ai"
    if confidence < midpoint:
        return "real"
    return "balanced"


def format_percent(value):
    return f"{value * 100:.1f}%"


def format_points(value):
    return f"{abs(value) * 100:.1f} pts"


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #050914;
            --bg-1: #081226;
            --bg-2: #0f1f3a;
            --panel: rgba(10, 19, 35, 0.76);
            --panel-strong: rgba(10, 22, 42, 0.92);
            --line: rgba(122, 207, 255, 0.18);
            --line-strong: rgba(122, 207, 255, 0.42);
            --text: #eef8ff;
            --muted: #9eb5ca;
            --cyan: #7ee8ff;
            --mint: #8dffd6;
            --rose: #ff8eae;
            --gold: #ffd79a;
            --real: #79f2c0;
            --ai: #ff8caa;
            --review: #ffd97a;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at 15% 15%, rgba(100, 195, 255, 0.20), transparent 22%),
                radial-gradient(circle at 85% 8%, rgba(255, 120, 175, 0.18), transparent 24%),
                radial-gradient(circle at 70% 78%, rgba(132, 255, 212, 0.12), transparent 20%),
                linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 42%, var(--bg-2) 100%);
            font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        section.main > div {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, p, label, div, span {
            color: var(--text);
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.8rem 1.5rem;
            border: 1px solid var(--line);
            background:
                radial-gradient(circle at 100% 0%, rgba(126, 232, 255, 0.14), transparent 24%),
                linear-gradient(135deg, rgba(9, 20, 39, 0.96), rgba(10, 22, 42, 0.70));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            animation: driftGlow 14s ease-in-out infinite alternate;
            margin-bottom: 1.25rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: -30% auto auto 68%;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(255, 163, 195, 0.20), transparent 68%);
            pointer-events: none;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.24em;
            font-size: 0.72rem;
            color: var(--cyan);
            margin-bottom: 0.65rem;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.02;
            letter-spacing: 0.02em;
        }

        .hero-copy {
            max-width: 48rem;
            margin: 0.9rem 0 1.1rem;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
        }

        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.58rem 0.9rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.03);
            color: var(--text);
            font-size: 0.9rem;
        }

        .section-note,
        .glass-panel {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 24px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(14px);
            margin-bottom: 1rem;
        }

        .section-note {
            margin: 1rem 0 0.6rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .verdict-card {
            border-radius: 26px;
            padding: 1.3rem 1.2rem 1.15rem;
            border: 1px solid var(--line);
            background:
                linear-gradient(160deg, rgba(10, 23, 44, 0.96), rgba(8, 17, 31, 0.82));
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.26);
            margin-bottom: 1rem;
        }

        .verdict-card.real {
            border-color: rgba(121, 242, 192, 0.35);
            box-shadow: 0 18px 60px rgba(46, 197, 150, 0.16);
        }

        .verdict-card.ai_generated {
            border-color: rgba(255, 140, 170, 0.38);
            box-shadow: 0 18px 60px rgba(231, 91, 134, 0.18);
        }

        .verdict-card.uncertain {
            border-color: rgba(255, 217, 122, 0.36);
            box-shadow: 0 18px 60px rgba(255, 198, 88, 0.16);
        }

        .panel-kicker {
            font-size: 0.74rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.7rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-size: 0.82rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .status-chip.real {
            background: rgba(121, 242, 192, 0.14);
            border: 1px solid rgba(121, 242, 192, 0.32);
            color: var(--real);
        }

        .status-chip.ai_generated {
            background: rgba(255, 140, 170, 0.13);
            border: 1px solid rgba(255, 140, 170, 0.33);
            color: var(--ai);
        }

        .status-chip.uncertain {
            background: rgba(255, 217, 122, 0.13);
            border: 1px solid rgba(255, 217, 122, 0.32);
            color: var(--review);
        }

        .verdict-headline {
            margin: 0;
            font-size: clamp(1.45rem, 2.4vw, 2.2rem);
            line-height: 1.1;
        }

        .verdict-copy {
            margin: 0.85rem 0 0;
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.98rem;
        }

        .signal-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .signal-card {
            border-radius: 22px;
            padding: 1rem 0.95rem;
            border: 1px solid rgba(255, 255, 255, 0.07);
            background: rgba(255, 255, 255, 0.035);
        }

        .signal-label {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .signal-value {
            display: block;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.15;
        }

        .signal-copy {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .subsection-title {
            margin: 0 0 0.8rem;
            font-size: 1.08rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--cyan);
        }

        .feature-list {
            margin: 0;
            padding-left: 1.1rem;
            color: var(--text);
        }

        .feature-list li {
            margin-bottom: 0.5rem;
            color: var(--text);
            line-height: 1.6;
        }

        .muted-copy {
            color: var(--muted);
            line-height: 1.65;
            margin: 0;
        }

        .callout {
            margin-top: 0.9rem;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            color: var(--muted);
            line-height: 1.65;
        }

        .upload-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1rem 0.8rem;
            background:
                radial-gradient(circle at 100% 0%, rgba(126, 232, 255, 0.11), transparent 24%),
                linear-gradient(160deg, rgba(9, 20, 39, 0.96), rgba(9, 16, 30, 0.78));
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
            margin-bottom: 0.9rem;
        }

        .upload-shell::after {
            content: "";
            position: absolute;
            inset: auto -10% -55% auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(126, 232, 255, 0.14), transparent 70%);
            pointer-events: none;
        }

        .upload-title {
            margin: 0 0 0.45rem;
            font-size: 1.05rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--cyan);
        }

        .upload-copy {
            margin: 0 0 0.95rem;
            color: var(--muted);
            line-height: 1.65;
        }

        .logic-shell {
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1.2rem;
            background:
                radial-gradient(circle at 100% 0%, rgba(255, 145, 214, 0.08), transparent 28%),
                linear-gradient(165deg, rgba(9, 19, 37, 0.98), rgba(8, 16, 30, 0.88));
            box-shadow: 0 16px 45px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
        }

        .logic-copy {
            margin: 0 0 0.8rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .logic-shell .feature-list {
            margin-bottom: 0;
        }

        .scan-ready-note {
            margin-top: 0.9rem;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(126, 232, 255, 0.14);
            background: rgba(255, 255, 255, 0.035);
            color: var(--muted);
            line-height: 1.6;
        }

        [data-testid="stFileUploader"] {
            width: 100%;
            margin-top: -0.15rem;
            margin-bottom: 0.35rem;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 22px !important;
            border: 1px dashed rgba(126, 232, 255, 0.34) !important;
            background:
                linear-gradient(160deg, rgba(12, 25, 47, 0.98), rgba(7, 15, 29, 0.90)) !important;
            box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.02),
                0 18px 45px rgba(0, 0, 0, 0.20);
            min-height: 220px;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(141, 255, 214, 0.48) !important;
            background:
                linear-gradient(160deg, rgba(16, 31, 57, 0.98), rgba(9, 18, 34, 0.92)) !important;
        }

        [data-testid="stFileUploaderDropzone"] > div {
            gap: 1rem;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            padding: 0.8rem 0.6rem;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--muted) !important;
        }

        [data-testid="stFileUploaderDropzone"] svg {
            fill: var(--cyan) !important;
        }

        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] section button[kind="secondary"],
        [data-testid="stFileUploader"] section button {
            border-radius: 999px !important;
            border: 1px solid rgba(126, 232, 255, 0.28) !important;
            background:
                linear-gradient(135deg, rgba(17, 34, 64, 0.98), rgba(10, 20, 39, 0.95)) !important;
            color: var(--text) !important;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18) !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploader"] section button[kind="secondary"]:hover,
        [data-testid="stFileUploader"] section button:hover {
            border-color: rgba(141, 255, 214, 0.34) !important;
            background:
                linear-gradient(135deg, rgba(23, 45, 82, 1), rgba(12, 25, 47, 0.98)) !important;
            color: var(--text) !important;
        }

        [data-testid="stFileUploaderDropzone"] button p,
        [data-testid="stFileUploader"] section button p,
        [data-testid="stFileUploaderDropzone"] button span,
        [data-testid="stFileUploader"] section button span {
            color: var(--text) !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(126, 232, 255, 0.18) !important;
            border-radius: 18px !important;
            color: var(--text) !important;
            width: calc(100% + 0.7rem) !important;
            margin-top: 0.65rem !important;
            margin-left: -0.35rem !important;
            margin-right: -0.35rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16);
        }

        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileData"] {
            color: var(--text) !important;
        }

        [data-testid="stFileUploaderDeleteBtn"] {
            color: var(--rose) !important;
        }

        [data-testid="stImage"],
        [data-testid="stImage"] > div,
        [data-testid="stImage"] > div > div,
        [data-testid="stPyplotChart"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.1rem;
            border-radius: 999px;
            border: 1px solid rgba(126, 232, 255, 0.34);
            background: linear-gradient(90deg, rgba(58, 140, 255, 0.85), rgba(110, 232, 255, 0.85));
            color: #04111f;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: 0 16px 40px rgba(55, 164, 255, 0.26);
        }

        .stButton > button:hover {
            border-color: rgba(141, 255, 214, 0.42);
            background: linear-gradient(90deg, rgba(110, 232, 255, 0.92), rgba(141, 255, 214, 0.92));
            color: #04111f;
        }

        .stImage img {
            border-radius: 24px;
            border: 1px solid var(--line);
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.28);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
        }

        [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }

        .glass-panel pre,
        .glass-panel code {
            white-space: pre-wrap !important;
            background: transparent !important;
            color: var(--text) !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            font-family: inherit !important;
            font-size: inherit !important;
        }

        .stAlert {
            border-radius: 18px;
        }

        @keyframes driftGlow {
            from {
                box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
                transform: translateY(0px);
            }
            to {
                box-shadow: 0 28px 88px rgba(0, 0, 0, 0.42);
                transform: translateY(-3px);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(input_size):
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Nebula Lens</div>
            <h1 class="hero-title">Galaxy-grade image forensics for Real vs AI detection</h1>
            <p class="hero-copy">
                Upload a frame and the app separates the <strong>final verdict</strong> from the raw model signal.
                Portraits automatically use a stricter AI cutoff to reduce false positives on real human photos.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">Input size {input_size} x {input_size}</span>
                <span class="hero-chip">Portrait safeguard enabled</span>
                <span class="hero-chip">Threshold-aware final verdicts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_verdict_content(final_decision, confidence, bounds, face_count):
    ai_signal = format_percent(confidence)
    ai_cutoff_text = format_percent(bounds["ai_cutoff"])
    real_floor_text = format_percent(bounds["real_floor"])
    review_lean = get_review_lean(confidence, bounds, face_count)

    if final_decision == "real":
        badge = "Likely Real"
        if face_count > 0 and bounds["ai_cutoff"] > bounds["real_floor"]:
            headline = "This portrait stayed well below the review band."
            narrative = (
                f"A face was detected, so the app used a portrait review band instead of forcing a quick decision. "
                f"The raw AI signal landed at <strong>{ai_signal}</strong>, which is clearly below the real floor of "
                f"<strong>{real_floor_text}</strong>, so the app keeps this result in the Real category."
            )
        else:
            headline = "This upload is being treated as a real photograph."
            narrative = (
                f"The model's raw AI signal stayed clearly below the active cutoff of "
                f"<strong>{ai_cutoff_text}</strong>, so the app keeps this result in the Real category."
            )
    elif final_decision == "ai_generated":
        badge = "Likely AI-Generated"
        if face_count > 0 and bounds["ai_cutoff"] > bounds["real_floor"]:
            headline = "This portrait cleared the higher AI cutoff."
            narrative = (
                f"A face was detected, so the app required a higher portrait AI cutoff of <strong>{ai_cutoff_text}</strong>. "
                f"The raw AI signal climbed to <strong>{ai_signal}</strong>, so the app still labels this upload as AI-generated."
            )
        else:
            headline = "This upload cleared the app's AI detection rule."
            narrative = (
                f"The raw AI signal reached <strong>{ai_signal}</strong>, which cleared the active cutoff of "
                f"<strong>{ai_cutoff_text}</strong>."
            )
    else:
        badge = "Needs Review"
        if face_count > 0 and bounds["ai_cutoff"] > bounds["real_floor"]:
            if review_lean == "ai":
                headline = "This portrait is still under review, but it leans AI."
                narrative = (
                    f"A face was detected, so the app used a portrait review band from <strong>{real_floor_text}</strong> "
                    f"to <strong>{ai_cutoff_text}</strong>. The raw AI signal landed at <strong>{ai_signal}</strong>, which is in the "
                    f"middle band but closer to the AI side, so the app avoids a hard label and marks this as review-needed."
                )
            elif review_lean == "real":
                headline = "This portrait is still under review, but it leans Real."
                narrative = (
                    f"A face was detected, so the app used a portrait review band from <strong>{real_floor_text}</strong> "
                    f"to <strong>{ai_cutoff_text}</strong>. The raw AI signal landed at <strong>{ai_signal}</strong>, which is in the "
                    f"middle band but closer to the Real side, so the app avoids a hard label and marks this as review-needed."
                )
            else:
                headline = "This portrait sits exactly in the middle review band."
                narrative = (
                    f"A face was detected, so the app used a portrait review band from <strong>{real_floor_text}</strong> "
                    f"to <strong>{ai_cutoff_text}</strong>. The raw AI signal landed at <strong>{ai_signal}</strong>, right around the "
                    f"middle of that band, so the app avoids forcing a hard Real or AI verdict."
                )
        else:
            headline = "This upload is too close to the decision boundary for a clean call."
            narrative = (
                f"The raw AI signal landed at <strong>{ai_signal}</strong>, too close to the active cutoff of "
                f"<strong>{ai_cutoff_text}</strong>, so the app recommends manual review."
            )

    return {
        "badge": badge,
        "headline": headline,
        "narrative": narrative,
    }


def render_verdict_card(final_decision, confidence, bounds, face_count):
    verdict = build_verdict_content(
        final_decision,
        confidence,
        bounds,
        face_count,
    )
    st.markdown(
        f"""
        <div class="verdict-card {final_decision}">
            <div class="panel-kicker">Final App Verdict</div>
            <div class="status-chip {final_decision}">{html.escape(verdict["badge"])}</div>
            <h2 class="verdict-headline">{html.escape(verdict["headline"])}</h2>
            <p class="verdict-copy">{verdict["narrative"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_grid(cards):
    card_html = "".join(
        f'<div class="signal-card">'
        f'<span class="signal-label">{html.escape(card["label"])}</span>'
        f'<span class="signal-value">{html.escape(card["value"])}</span>'
        f'<div class="signal-copy">{html.escape(card["copy"])}</div>'
        f"</div>"
        for card in cards
    )
    st.markdown(f'<div class="signal-grid">{card_html}</div>', unsafe_allow_html=True)


def render_panel(title, body_html):
    cleaned_body_html = textwrap.dedent(body_html).strip()
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="subsection-title">{html.escape(title)}</div>
            {cleaned_body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_gradcam_heatmap(img_array, model):
    try:
        # Convert to TensorFlow variable to ensure gradients work
        img_tensor = tf.Variable(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = model(img_tensor)
            loss = predictions[:, 0]
        
        # Get gradients
        grads = tape.gradient(loss, img_tensor)
        
        if grads is None:
            return None
            
        # Process gradients
        grads = tf.abs(grads)
        grads = tf.reduce_mean(grads, axis=-1)
        grads = grads[0]  # Remove batch dimension
        
        # Normalize
        grads_min = tf.reduce_min(grads)
        grads_max = tf.reduce_max(grads)
        if grads_max > grads_min:
            grads = (grads - grads_min) / (grads_max - grads_min)
        
        return grads.numpy()
    except:
        return None


def describe_image_content(image):
    try:
        # Ensure RGB format for content analysis
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        content_model = load_content_model()
        predictions = content_model.predict(img_array, verbose=0)
        decoded = decode_predictions(predictions, top=3)[0]
        
        # Create descriptive paragraph with better filtering
        items = [label.replace('_', ' ').lower() for _, label, prob in decoded if prob > 0.1]
        
        if len(items) >= 2:
            description = f"This image shows {items[0]}, and possibly {items[1]}"
            if len(items) > 2:
                description += f" or {items[2]}"
        elif len(items) == 1:
            description = f"This image shows {items[0]}"
        else:
            # Show top prediction even if confidence is low
            top_item = decoded[0][1].replace('_', ' ').lower()
            description = f"This image might contain {top_item}, though the model is uncertain"
            
        return description + "."
    except Exception as e:
        return "Unable to analyze image content."


model = load_model()
MODEL_INPUT_SIZE = get_model_image_size(model)

inject_styles()
render_hero(MODEL_INPUT_SIZE)
analyze_clicked = False
control_left, control_right = st.columns([1.42, 0.78], gap="large")

with control_left:
    st.markdown(
        """
        <div class="upload-shell">
            <div class="upload-title">Drop An Image Into The Scan Deck</div>
            <p class="upload-copy">
                Upload a portrait, scene, render, screenshot, or photo. The interface will show the final verdict first,
                then the raw model signal and supporting evidence below it.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    img = st.file_uploader(
        "Upload an image to inspect",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("Run Nebula Scan", disabled=img is None)
    if img and not analyze_clicked:
        st.markdown(
            """
            <div class="scan-ready-note">
                The frame is loaded into the scan deck. Press <strong>Run Nebula Scan</strong> to reveal the final verdict,
                the raw model signal, and the visual focus map below.
            </div>
            """,
            unsafe_allow_html=True,
        )

with control_right:
    st.markdown(
        """
        <div class="logic-shell">
            <div class="subsection-title">Scan Logic</div>
            <p class="logic-copy">
                The upload controls live together on the left. This panel is just a quick decoder for how the app turns
                the model signal into the final verdict.
            </p>
            <ul class="feature-list">
                <li><strong>Real</strong> means the upload stayed below the active AI cutoff.</li>
                <li><strong>AI Generated</strong> means the upload cleared the cutoff with enough margin.</li>
                <li><strong>Needs Review</strong> means the upload is too close to the boundary for a clean call.</li>
                <li><strong>Portraits</strong> use a stricter AI threshold to reduce false positives on real faces.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if img and analyze_clicked:
    original_image = Image.open(img)
    image = original_image.copy()
    original_size = image.size
    original_mode = image.mode
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    face_count = detect_face_count(image)
    decision_bounds = get_decision_bounds(face_count)
    
    image = ImageOps.fit(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.LANCZOS)
    img_array = img_to_array(image).astype("float32")
    if MODEL_INPUT_SIZE == LEGACY_IMG_SIZE:
        img_array = img_array / 255.0
    test = np.array([img_array])
    
    # Run prediction first to build the model
    y = model.predict(test)
    confidence = float(y[0][0])
    final_decision = get_final_decision(confidence, decision_bounds, face_count)
    raw_ai_score = format_percent(confidence)
    ai_cutoff_text = format_percent(decision_bounds["ai_cutoff"])
    real_floor_text = format_percent(decision_bounds["real_floor"])
    review_lean = get_review_lean(confidence, decision_bounds, face_count)
    threshold_gap_text = get_threshold_gap_text(
        confidence,
        decision_bounds,
        face_count,
        final_decision,
    )

    decision_mode = "Portrait review band" if face_count > 0 else "Standard mode"
    signal_cards = [
        {
            "label": "Model AI Signal",
            "value": raw_ai_score,
            "copy": "Raw neural-network output, not the final verdict.",
        },
    ]
    if face_count > 0 and decision_bounds["ai_cutoff"] > decision_bounds["real_floor"]:
        signal_cards.extend(
            [
                {
                    "label": "Portrait AI Cutoff",
                    "value": ai_cutoff_text,
                    "copy": "Face images need to clear this higher bar for a hard AI verdict.",
                },
                {
                    "label": "Real Floor",
                    "value": real_floor_text,
                    "copy": "Face images only become hard Real when the raw AI signal is clearly low.",
                },
                {
                    "label": "Decision Zone",
                    "value": (
                        "AI-leaning review"
                        if final_decision == "uncertain" and review_lean == "ai"
                        else "Real-leaning review"
                        if final_decision == "uncertain" and review_lean == "real"
                        else threshold_gap_text
                    ),
                    "copy": (
                        "This portrait is still in the middle band, but the raw score sits closer to the AI side."
                        if final_decision == "uncertain" and review_lean == "ai"
                        else "This portrait is still in the middle band, but the raw score sits closer to the Real side."
                        if final_decision == "uncertain" and review_lean == "real"
                        else "Scores in the middle band are treated as Needs Review instead of forced Real."
                    ),
                },
                {
                    "label": "Decision Mode",
                    "value": decision_mode,
                    "copy": "Portraits use a wider middle band to reduce both false AI and false Real calls.",
                },
            ]
        )
    else:
        signal_cards.extend(
            [
                {
                    "label": "Active Threshold",
                    "value": ai_cutoff_text,
                    "copy": "The cutoff currently required for an AI label.",
                },
                {
                    "label": "Threshold Gap",
                    "value": threshold_gap_text,
                    "copy": "How far the raw AI signal sits from the cutoff.",
                },
                {
                    "label": "Decision Mode",
                    "value": decision_mode,
                    "copy": "Standard scenes use a single cutoff with a small review band.",
                },
            ]
        )

    top_left, top_right = st.columns([1.05, 1.2], gap="large")
    with top_left:
        render_panel(
            "Uploaded Frame",
            f"""
            <p class="muted-copy">
                Source size <strong>{original_size[0]} x {original_size[1]}</strong> pixels.
                The classifier resizes it to <strong>{MODEL_INPUT_SIZE} x {MODEL_INPUT_SIZE}</strong> for analysis.
            </p>
            """,
        )
        st.image(original_image, use_container_width=True)

    with top_right:
        render_verdict_card(
            final_decision,
            confidence,
            decision_bounds,
            face_count,
        )
        render_signal_grid(signal_cards)
        render_panel(
            "How To Read This Result",
            """
            <ul class="feature-list">
                <li><strong>Final app verdict</strong> is the answer the interface wants you to trust.</li>
                <li><strong>Model AI signal</strong> is only the raw model tendency score.</li>
                <li><strong>Portraits</strong> use a wider review band: clearly low can be Real, clearly high can be AI, and the middle zone becomes Needs Review.</li>
                <li><strong>Scene guess</strong> below comes from a separate model and does not decide Real vs AI.</li>
            </ul>
            """,
        )

    forensic_notes = [
        f"Color mode: {original_mode}",
        f"Decision mode: {decision_mode}",
        f"Raw AI signal: {raw_ai_score}",
    ]
    if face_count > 0:
        forensic_notes.append(
            f"Portrait safeguard: {face_count} face(s) detected, so the app used a review band from {real_floor_text} to {ai_cutoff_text}"
        )
        forensic_notes.append(f"Portrait AI cutoff: {ai_cutoff_text}")
        forensic_notes.append(f"Real floor: {real_floor_text}")
    else:
        forensic_notes.append(f"Active AI threshold: {ai_cutoff_text}")
    if final_decision == "real":
        forensic_notes.append(
            f"Final call: the raw AI signal finished {threshold_gap_text}, so the app kept this in the Real category"
        )
    elif final_decision == "ai_generated":
        forensic_notes.append(
            f"Final call: the raw AI signal finished {threshold_gap_text}, so the app labeled it as AI-generated"
        )
    else:
        forensic_notes.append(
            f"Final call: the raw AI signal landed in the review band, so the app asked for manual review"
        )
        if review_lean == "ai":
            forensic_notes.append(
                "Review lean: the score sits closer to the AI side of the portrait review band"
            )
        elif review_lean == "real":
            forensic_notes.append(
                "Review lean: the score sits closer to the Real side of the portrait review band"
            )

    details_left, details_right = st.columns([1, 1.05], gap="large")
    with details_left:
        notes_html = "".join(
            f"<li>{html.escape(item)}</li>" for item in forensic_notes
        )
        render_panel(
            "Forensic Notes",
            f'<ul class="feature-list">{notes_html}</ul>',
        )

        content_description = describe_image_content(original_image)
        render_panel(
            "Scene Guess",
            f"""
            <p class="muted-copy">
                {html.escape(content_description)}
            </p>
            <div class="callout">
                This scene guess is produced by a separate ImageNet recognition model.
                It is helpful for context, but it does <strong>not</strong> decide whether the upload is Real or AI-generated.
            </div>
            """,
        )

    with details_right:
        render_panel(
            "Model Focus Map",
            """
            <p class="muted-copy">
                The heatmap below belongs to the Real-vs-AI classifier.
                Hotter regions show where the classifier concentrated most while forming its verdict.
            </p>
            """,
        )
        heatmap = make_gradcam_heatmap(test, model)
        if heatmap is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6))
            fig.patch.set_facecolor("#081226")

            ax1.imshow(test[0])
            ax1.set_title("Analyzed Frame", color="#eef8ff", fontsize=11)
            ax1.axis('off')
            ax1.set_facecolor("#081226")

            ax2.imshow(test[0])
            ax2.imshow(heatmap, alpha=0.58, cmap='magma')
            ax2.set_title("Classifier Focus", color="#eef8ff", fontsize=11)
            ax2.axis('off')
            ax2.set_facecolor("#081226")

            st.pyplot(fig, use_container_width=True)
        else:
            render_panel(
                "Model Focus Map",
                """
                <p class="muted-copy">
                    A visual focus map could not be generated for this prediction.
                </p>
                """,
            )
