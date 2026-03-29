import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model_utils import (
    DEFAULT_IMG_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    create_transfer_learning_model,
)


ROOT_DIR = Path(__file__).resolve().parent
REAL_DIR = ROOT_DIR / "real"
AI_DIR = ROOT_DIR / "ai_generated"
DEFAULT_MANIFEST_PATH = ROOT_DIR / "datasets" / "dataset_manifest.json"
CHECKPOINT_PATH = ROOT_DIR / "AIGeneratedModel.keras"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_INT = {
    "real": 0,
    "ai_generated": 1,
    "ai-generated": 1,
    "ai": 1,
    "fake": 1,
    "synthetic": 1,
}
INT_TO_LABEL = {0: "real", 1: "ai_generated"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AI image classifier.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to a dataset manifest. Falls back to local real/ and ai_generated/ folders when missing.",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet weights for transfer learning when available.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=("none", "class", "source_class"),
        default="source_class",
        help="Rebalance the training split by class or by source/class before fitting.",
    )
    parser.add_argument(
        "--balance-target-per-group",
        type=int,
        default=0,
        help="Exact sample target per balance group. Use 0 to pick an automatic target.",
    )
    parser.add_argument(
        "--balance-cap-per-group",
        type=int,
        default=4000,
        help="Upper cap for the automatic balance target per group.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export the existing best .keras checkpoint to .weights.h5.",
    )
    return parser.parse_args()


def normalize_label(label):
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        value = int(label)
        if value in INT_TO_LABEL:
            return value
    normalized = str(label).strip().lower().replace(" ", "_")
    if normalized in LABEL_TO_INT:
        return LABEL_TO_INT[normalized]
    raise ValueError(f"Unsupported label: {label}")


def make_record(path, label, source):
    return {
        "path": Path(path).expanduser().resolve(),
        "label": normalize_label(label),
        "source": str(source),
    }


def list_image_files(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_manifest_samples(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    samples = []
    missing = 0

    for item in manifest.get("samples", []):
        record = make_record(
            item["path"],
            item["label"],
            item.get("source", "unknown_source"),
        )
        if record["path"].is_file():
            samples.append(record)
        else:
            missing += 1

    if not samples:
        raise FileNotFoundError(
            f"No usable image files were found in manifest: {manifest_path}"
        )
    if missing:
        print(f"Skipped {missing} manifest entries because the files were missing on disk.")

    return samples


def load_local_samples():
    real_paths = list_image_files(REAL_DIR)
    ai_paths = list_image_files(AI_DIR)
    if not real_paths or not ai_paths:
        raise FileNotFoundError(
            "Neither a manifest nor usable local real/ and ai_generated/ folders were found."
        )

    return [make_record(path, 0, "local_repo") for path in real_paths] + [
        make_record(path, 1, "local_repo") for path in ai_paths
    ]


def compute_split_sizes(total):
    if total < 3:
        raise ValueError(
            f"Each source/class group needs at least 3 images for train/val/test splitting. Got {total}."
        )

    train_count = max(1, int(total * 0.7))
    val_count = max(1, int(total * 0.15))
    test_count = total - train_count - val_count

    while test_count < 1:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
        test_count = total - train_count - val_count

    if train_count < 1 or val_count < 1 or test_count < 1:
        raise ValueError(
            f"Could not create valid train/validation/test splits for a group of size {total}."
        )

    return train_count, val_count, test_count


def shuffle_records(records, seed):
    if not records:
        return []
    indices = tf.random.shuffle(tf.range(len(records)), seed=seed).numpy().tolist()
    return [records[index] for index in indices]


def split_group(records, seed):
    shuffled = shuffle_records(records, seed)

    train_count, val_count, _ = compute_split_sizes(len(shuffled))
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def split_records(records, seed):
    groups = defaultdict(list)
    for record in records:
        groups[(record["source"], record["label"])].append(record)

    splits = {"train": [], "val": [], "test": []}
    for offset, group_key in enumerate(sorted(groups)):
        train_records, val_records, test_records = split_group(
            groups[group_key],
            seed + offset,
        )
        splits["train"].extend(train_records)
        splits["val"].extend(val_records)
        splits["test"].extend(test_records)

    return splits


def make_balance_group_key(record, balance_mode):
    label_name = INT_TO_LABEL[record["label"]]
    if balance_mode == "class":
        return label_name
    if balance_mode == "source_class":
        return f"{record['source']}::{label_name}"
    return "all_samples"


def choose_balance_target(group_sizes, explicit_target, cap_target):
    if explicit_target > 0:
        return explicit_target

    smallest_group = min(group_sizes)
    largest_group = max(group_sizes)
    if largest_group <= smallest_group * 2:
        return largest_group

    median_group = int(median(group_sizes))
    auto_target = max(smallest_group * 4, median_group * 3)
    if cap_target > 0:
        auto_target = min(auto_target, cap_target)
    return max(1, auto_target)


def resample_group(records, target_count, seed):
    shuffled = shuffle_records(records, seed)
    if len(shuffled) >= target_count:
        return shuffled[:target_count]

    repeats, remainder = divmod(target_count, len(shuffled))
    sampled = shuffled * repeats
    if remainder:
        sampled.extend(shuffle_records(shuffled, seed + 1000)[:remainder])
    return shuffle_records(sampled, seed + 2000)


def rebalance_training_records(records, balance_mode, seed, explicit_target, cap_target):
    if balance_mode == "none":
        return list(records), {
            "mode": "none",
            "sample_count_before": len(records),
            "sample_count_after": len(records),
        }

    grouped = defaultdict(list)
    for record in records:
        grouped[make_balance_group_key(record, balance_mode)].append(record)

    group_counts_before = {
        group_key: len(grouped[group_key]) for group_key in sorted(grouped)
    }
    target_count = choose_balance_target(
        list(group_counts_before.values()),
        explicit_target=explicit_target,
        cap_target=cap_target,
    )

    balanced_records = []
    group_counts_after = {}
    for offset, group_key in enumerate(sorted(grouped)):
        sampled_group = resample_group(
            grouped[group_key],
            target_count=target_count,
            seed=seed + offset,
        )
        balanced_records.extend(sampled_group)
        group_counts_after[group_key] = len(sampled_group)

    balanced_records = shuffle_records(balanced_records, seed + 999)
    return balanced_records, {
        "mode": balance_mode,
        "target_per_group": target_count,
        "sample_count_before": len(records),
        "sample_count_after": len(balanced_records),
        "group_counts_before": group_counts_before,
        "group_counts_after": group_counts_after,
    }


def decode_and_resize(path, label, img_size):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def build_dataset(samples, img_size, batch_size, training, seed):
    paths = [str(sample["path"]) for sample in samples]
    labels = [sample["label"] for sample in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: decode_and_resize(path, label, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def predict_scores(model, samples, img_size, batch_size, seed):
    dataset = build_dataset(
        samples,
        img_size=img_size,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )
    scores = model.predict(dataset, verbose=0).reshape(-1)
    labels = np.array([sample["label"] for sample in samples], dtype=np.int32)
    return labels, scores


def safe_divide(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def compute_threshold_metrics(labels, scores, threshold):
    threshold = float(threshold)
    predictions = (scores >= threshold).astype(np.int32)

    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))

    recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)

    return {
        "threshold": threshold,
        "accuracy": safe_divide(true_positive + true_negative, len(labels)),
        "precision": safe_divide(true_positive, true_positive + false_positive),
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": safe_divide(false_positive, false_positive + true_negative),
        "false_negative_rate": safe_divide(false_negative, false_negative + true_positive),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def load_existing_summary():
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def count_labels(records):
    counts = {"real": 0, "ai_generated": 0}
    for record in records:
        counts[INT_TO_LABEL[record["label"]]] += 1
    return counts


def count_by_source(records):
    counts = {}
    grouped = defaultdict(lambda: {"real": 0, "ai_generated": 0})
    for record in records:
        grouped[record["source"]][INT_TO_LABEL[record["label"]]] += 1
    for source_name in sorted(grouped):
        counts[source_name] = grouped[source_name]
    return counts


def compute_threshold_metrics_by_source(samples, scores, threshold):
    grouped_labels = defaultdict(list)
    grouped_scores = defaultdict(list)
    for sample, score in zip(samples, scores):
        grouped_labels[sample["source"]].append(sample["label"])
        grouped_scores[sample["source"]].append(float(score))

    metrics = {}
    for source_name in sorted(grouped_labels):
        labels = np.array(grouped_labels[source_name], dtype=np.int32)
        source_scores = np.array(grouped_scores[source_name], dtype=np.float32)
        metrics[source_name] = compute_threshold_metrics(
            labels,
            source_scores,
            threshold,
        )
    return metrics


def average_source_metric(metrics_by_source, key):
    if not metrics_by_source:
        return 0.0
    return float(np.mean([metrics[key] for metrics in metrics_by_source.values()]))


def select_decision_threshold(samples, labels, scores):
    best_metrics = None
    best_key = None
    for threshold in np.arange(0.2, 0.91, 0.01):
        metrics = compute_threshold_metrics(labels, scores, threshold)
        metrics_by_source = compute_threshold_metrics_by_source(
            samples,
            scores,
            threshold,
        )
        metrics["macro_balanced_accuracy"] = average_source_metric(
            metrics_by_source,
            "balanced_accuracy",
        )
        metrics["macro_specificity"] = average_source_metric(
            metrics_by_source,
            "specificity",
        )
        metrics["macro_recall"] = average_source_metric(metrics_by_source, "recall")
        candidate_key = (
            metrics["macro_balanced_accuracy"],
            metrics["macro_specificity"],
            metrics["balanced_accuracy"],
            metrics["specificity"],
            metrics["precision"],
            metrics["threshold"],
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_metrics = metrics

    return best_metrics


def evaluate_by_source(model, samples, img_size, batch_size, seed):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["source"]].append(sample)

    metrics = {}
    for source_name in sorted(grouped):
        dataset = build_dataset(
            grouped[source_name],
            img_size=img_size,
            batch_size=batch_size,
            training=False,
            seed=seed,
        )
        result = model.evaluate(dataset, return_dict=True, verbose=0)
        metrics[source_name] = {key: float(value) for key, value in result.items()}
    return metrics


def export_inference_model(best_model, img_size):
    inference_model, _ = create_transfer_learning_model(
        img_size=img_size,
        pretrained=False,
        augment=False,
    )
    inference_model.set_weights(best_model.get_weights())
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    inference_model.save_weights(MODEL_PATH)
    return inference_model


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        all_samples = load_manifest_samples(manifest_path)
        data_source = str(manifest_path)
        print(f"Using dataset manifest: {manifest_path}")
    else:
        all_samples = load_local_samples()
        data_source = "local_folders"
        print("Manifest not found. Falling back to local real/ and ai_generated/ folders.")

    splits = split_records(all_samples, args.seed)

    effective_train_records, train_sampling = rebalance_training_records(
        splits["train"],
        balance_mode=args.balance_mode,
        seed=args.seed,
        explicit_target=args.balance_target_per_group,
        cap_target=args.balance_cap_per_group,
    )

    train_dataset = build_dataset(
        effective_train_records,
        args.img_size,
        args.batch_size,
        training=True,
        seed=args.seed,
    )
    val_dataset = build_dataset(
        splits["val"], args.img_size, args.batch_size, training=False, seed=args.seed
    )
    test_dataset = build_dataset(
        splits["test"], args.img_size, args.batch_size, training=False, seed=args.seed
    )

    history_data = {}
    if not args.export_only:
        use_pretrained = args.weights == "imagenet"
        model, base_model = create_transfer_learning_model(
            img_size=args.img_size,
            pretrained=use_pretrained,
            augment=True,
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(CHECKPOINT_PATH),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        compile_model(model, learning_rate=1e-3)
        head_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.head_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune_layers]:
            layer.trainable = False

        compile_model(model, learning_rate=1e-5)
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.head_epochs + args.fine_tune_epochs,
            initial_epoch=head_history.epoch[-1] + 1,
            callbacks=callbacks,
            verbose=1,
        )
        history_data = merge_histories(head_history, fine_tune_history)
    elif not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "AIGeneratedModel.keras was not found, so export-only mode cannot run."
        )

    best_model = keras.models.load_model(CHECKPOINT_PATH, compile=False)
    compile_model(best_model, learning_rate=1e-5)
    test_metrics = best_model.evaluate(test_dataset, return_dict=True, verbose=1)
    test_metrics_by_source = evaluate_by_source(
        best_model,
        splits["test"],
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    val_labels, val_scores = predict_scores(
        best_model,
        splits["val"],
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    decision_threshold_summary = select_decision_threshold(
        splits["val"],
        val_labels,
        val_scores,
    )
    decision_threshold = decision_threshold_summary["threshold"]
    test_labels, test_scores = predict_scores(
        best_model,
        splits["test"],
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    test_threshold_metrics = compute_threshold_metrics(
        test_labels,
        test_scores,
        decision_threshold,
    )
    test_threshold_metrics_by_source = compute_threshold_metrics_by_source(
        splits["test"],
        test_scores,
        decision_threshold,
    )
    export_inference_model(best_model, args.img_size)

    summary = load_existing_summary() if args.export_only else {}
    summary.update(
        {
            "image_size": summary.get("image_size", args.img_size)
            if args.export_only
            else args.img_size,
            "weights": summary.get("weights", args.weights)
            if args.export_only
            else args.weights,
            "data_source": data_source,
            "class_counts": count_labels(all_samples),
            "source_counts": count_by_source(all_samples),
            "split_counts": {
                split_name: count_labels(split_records_list)
                for split_name, split_records_list in splits.items()
            },
            "effective_train_counts": count_labels(effective_train_records),
            "split_source_counts": {
                split_name: count_by_source(split_records_list)
                for split_name, split_records_list in splits.items()
            },
            "effective_train_source_counts": count_by_source(effective_train_records),
            "train_sampling": train_sampling,
            "history": summary.get("history", {}) if args.export_only else history_data,
            "test_metrics": {key: float(value) for key, value in test_metrics.items()},
            "test_metrics_by_source": test_metrics_by_source,
            "decision_threshold": float(decision_threshold),
            "decision_threshold_selection": decision_threshold_summary,
            "test_threshold_metrics": test_threshold_metrics,
            "test_threshold_metrics_by_source": test_threshold_metrics_by_source,
        }
    )
    METRICS_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved training metrics to {METRICS_PATH}")
    print(json.dumps(summary["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
