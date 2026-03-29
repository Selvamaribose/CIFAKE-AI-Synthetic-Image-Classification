import argparse
import json
from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "dataset_sources.json"
DEFAULT_DATASETS_DIR = ROOT_DIR / "datasets"
DEFAULT_INDEX_PATH = DEFAULT_DATASETS_DIR / "source_index.json"
DEFAULT_MANIFEST_PATH = DEFAULT_DATASETS_DIR / "dataset_manifest.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_REAL_KEYWORDS = [
    "real",
    "natural",
    "photo",
    "photos",
    "photograph",
    "photographs",
    "authentic",
    "human",
]
DEFAULT_AI_KEYWORDS = [
    "ai",
    "generated",
    "synthetic",
    "fake",
    "artificial",
    "computer_generated",
    "computer-generated",
    "cg",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets and build a source-aware image manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional source names from dataset_sources.json to process.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download each configured dataset with kagglehub before building the manifest.",
    )
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Scan resolved dataset folders and build a manifest JSON file.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip sources that fail to resolve or download instead of stopping the whole run.",
    )
    return parser.parse_args()


def normalize_text(value):
    return str(value).lower().replace("-", "_").replace(" ", "_")


def tokenize_text(value):
    raw_value = str(value).replace("-", "_").replace(" ", "_")
    camel_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw_value)
    normalized = normalize_text(camel_spaced)
    cleaned = re.sub(r"[^a-z0-9_]+", "_", normalized)
    return {token for token in cleaned.split("_") if token}


def has_keyword_match(path_parts, keyword):
    normalized_keyword = normalize_text(keyword)
    keyword_tokens = [token for token in normalized_keyword.split("_") if token]
    normalized_parts = [normalize_text(part) for part in path_parts]
    part_tokens = set()
    for part in path_parts:
        part_tokens.update(tokenize_text(part))

    if normalized_keyword in normalized_parts:
        return True
    if len(keyword_tokens) == 1:
        return keyword_tokens[0] in part_tokens
    return all(token in part_tokens for token in keyword_tokens)


def load_sources(config_path):
    config = json.loads(config_path.read_text())
    sources = config.get("sources", config)
    if not sources:
        raise ValueError(f"No dataset sources were defined in {config_path}")
    return sources


def filter_sources(sources, selected_names):
    if not selected_names:
        return sources

    selected = set(selected_names)
    filtered = [source for source in sources if source["name"] in selected]
    if not filtered:
        raise ValueError(
            "None of the requested --sources names were found in dataset_sources.json."
        )
    return filtered


def load_index(index_path):
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {item["name"]: item for item in payload.get("sources", [])}


def save_index(index_path, entries):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sources": entries}
    index_path.write_text(json.dumps(payload, indent=2))


def resolve_sources(sources, existing_index, should_download, continue_on_error):
    resolved = []

    if should_download:
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for --download. Install it with `pip install kagglehub`."
            ) from exc

    for source in sources:
        name = source["name"]
        dataset_path = None

        try:
            if "local_path" in source:
                dataset_path = Path(source["local_path"]).expanduser().resolve()
            elif name in existing_index:
                candidate = Path(existing_index[name]["local_path"]).expanduser().resolve()
                if candidate.exists():
                    dataset_path = candidate

            if dataset_path is None and should_download:
                dataset_path = Path(kagglehub.dataset_download(source["kaggle_id"])).resolve()

            if dataset_path is None:
                raise FileNotFoundError(
                    f"Could not resolve a local path for source '{name}'. "
                    f"Run with --download or provide local_path in {DEFAULT_CONFIG_PATH.name}."
                )

            resolved.append(
                {
                    "name": name,
                    "kaggle_id": source["kaggle_id"],
                    "local_path": str(dataset_path),
                    "real_keywords": source.get("real_keywords", []),
                    "ai_keywords": source.get("ai_keywords", []),
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"Skipping {name}: {exc}")

    if not resolved:
        raise FileNotFoundError("No dataset sources were resolved successfully.")

    return resolved


def infer_label(relative_path, source):
    real_keywords = DEFAULT_REAL_KEYWORDS + source.get("real_keywords", [])
    ai_keywords = DEFAULT_AI_KEYWORDS + source.get("ai_keywords", [])

    directory_parts = relative_path.parts[:-1]
    file_parts = (relative_path.stem,)

    real_dir = any(has_keyword_match(directory_parts, keyword) for keyword in real_keywords)
    ai_dir = any(has_keyword_match(directory_parts, keyword) for keyword in ai_keywords)
    if real_dir and not ai_dir:
        return "real"
    if ai_dir and not real_dir:
        return "ai_generated"

    real_file = any(has_keyword_match(file_parts, keyword) for keyword in real_keywords)
    ai_file = any(has_keyword_match(file_parts, keyword) for keyword in ai_keywords)
    if real_file and not ai_file:
        return "real"
    if ai_file and not real_file:
        return "ai_generated"

    return None


def scan_source(source):
    root = Path(source["local_path"]).resolve()
    samples = []
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = path.relative_to(root)
        label = infer_label(relative_path, source)
        if label is None:
            skipped += 1
            continue

        samples.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "source": source["name"],
                "kaggle_id": source["kaggle_id"],
            }
        )

    return samples, skipped


def build_manifest(manifest_path, resolved_sources):
    all_samples = []
    summary = {}
    skipped_counts = {}

    for source in resolved_sources:
        samples, skipped = scan_source(source)
        counts = {"real": 0, "ai_generated": 0}
        for sample in samples:
            counts[sample["label"]] += 1

        summary[source["name"]] = counts
        skipped_counts[source["name"]] = skipped
        all_samples.extend(samples)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sources": [
            {
                "name": source["name"],
                "kaggle_id": source["kaggle_id"],
                "local_path": source["local_path"],
                "counts": summary[source["name"]],
                "skipped_unlabeled": skipped_counts[source["name"]],
            }
            for source in resolved_sources
        ],
        "samples": all_samples,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved dataset manifest to {manifest_path}")
    for source_name in sorted(summary):
        counts = summary[source_name]
        print(
            f"{source_name}: real={counts['real']}, ai_generated={counts['ai_generated']}, "
            f"skipped={skipped_counts[source_name]}"
        )


def main():
    args = parse_args()
    datasets_dir = args.datasets_dir.resolve()
    datasets_dir.mkdir(parents=True, exist_ok=True)

    should_download = args.download
    should_build_manifest = args.build_manifest
    if not should_download and not should_build_manifest:
        should_download = True
        should_build_manifest = True

    index_path = args.index.resolve()
    manifest_path = args.manifest.resolve()
    if args.index == DEFAULT_INDEX_PATH:
        index_path = datasets_dir / "source_index.json"
    if args.manifest == DEFAULT_MANIFEST_PATH:
        manifest_path = datasets_dir / "dataset_manifest.json"

    sources = load_sources(args.config.resolve())
    sources = filter_sources(sources, args.sources)
    existing_index = load_index(index_path)
    resolved_sources = resolve_sources(
        sources,
        existing_index,
        should_download,
        continue_on_error=args.continue_on_error,
    )
    save_index(index_path, resolved_sources)
    print(f"Saved source index to {index_path}")

    if should_build_manifest:
        build_manifest(manifest_path, resolved_sources)


if __name__ == "__main__":
    main()
