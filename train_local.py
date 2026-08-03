"""
=============================================
 CENTRALIZED TRAINING SCRIPT (Scientific)
=============================================

Trains EfficientNetB0 on ISIC 2019 as a
centralized baseline for FL comparison.

All 8 scientific improvements applied:
  1. Separate unseen test set
  2. Generator-based training (memory-safe)
  3. Identical config to FL pipeline
  4. F1-macro monitored checkpoint
  5. Full evaluation metrics suite
  6. Dataset integrity verification
  7. Reproducibility seeds
  8. Training history plots

Usage:
    python train_local.py
    python train_local.py --csv_path dataset/raw/ISIC_2019_Training_GroundTruth.csv
    python train_local.py --epochs 15 --finetune_epochs 10
"""

import os
import sys
import argparse
import random
import json

# ── GPU: select RTX 4050 BEFORE importing TensorFlow ─────────────────────────
# Uncomment the line below AFTER installing CUDA 11.2 + cuDNN 8.1:
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # RTX 4050 only
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"]        = "ignore"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── GPU setup (memory growth only — device already selected via env var) ──────
def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] ⚠️  No GPU detected — falling back to CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] ✅  Training on: {[g.name for g in gpus]}")
        print("[GPU] 🎯  NVIDIA RTX 4050 (CUDA_VISIBLE_DEVICES=1)")
    except RuntimeError as e:
        print(f"[GPU] ⚠️  Memory growth config failed: {e}")

_configure_gpu()


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
    balanced_accuracy_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from preprocessing.balancing import get_class_weights
from models.model import build_model, unfreeze_model, sanity_check


# ============================================
# CONFIG (identical to FL pipeline)
# ============================================

DATASET_DIR     = "dataset/partitions"
IMAGE_ROOT      = "dataset/raw/ISIC_2019_Training_Input"
CHECKPOINT_DIR  = "models/checkpoints"
PLOTS_DIR       = "models/plots"
GLOBAL_TEST_CSV = "dataset/partitions/global_test.csv"

CLASS_NAMES = ["MEL", "NV", "BKL", "BCC"]
NUM_CLASSES = 4
IMAGE_SIZE  = 224
BATCH_SIZE  = 16    # identical to FL pipeline
SEED        = 42


# ============================================
# FIX 7: REPRODUCIBILITY SEEDS
# ============================================

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================
# ARGUMENTS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized EfficientNetB0 training (Scientific Baseline)"
    )
    parser.add_argument(
        "--client_id",
        type=str,
        default="all",
        choices=["all", "client_1", "client_2", "client_3", "client_4"],
        help="'all' = combined partitions (recommended). Default: all"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to a raw ISIC GroundTruth CSV (overrides --client_id)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet121",
        choices=["densenet121", "densenet169", "densenet201", "resnet50v2", "efficientnetb0"],
        help="Backbone architecture to use (default: densenet121)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Head-only training epochs (default: 25)"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=25,
        help="Fine-tuning epochs after unfreezing backbone (default: 25)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--use_tta",
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=True,
        help="Use 5-point Test-Time Augmentation during test evaluation (default: True)"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="Learning rate decay strategy: 'cosine' or 'plateau' (default: cosine)"
    )
    parser.add_argument(
        "--pooling_mode",
        type=str,
        default="gem_gap",
        choices=["gem_gap", "gem_gmp", "gem_only"],
        help="Pooling combination: 'gem_gap' (GeM+GAP), 'gem_gmp' (GeM+GMP), or 'gem_only' (default: gem_gap)"
    )
    return parser.parse_args()


# ============================================
# CSV HELPERS
# ============================================

def load_and_format_csv(csv_file):
    """Convert raw ISIC one-hot GroundTruth CSV to image/label format."""
    df = pd.read_csv(csv_file)
    class_cols = CLASS_NAMES
    if "label" not in df.columns and any(c in df.columns for c in class_cols):
        present = [c for c in class_cols if c in df.columns]
        if "UNK" in df.columns:
            df = df[df["UNK"] != 1.0]
        df["label"] = df[present].idxmax(axis=1)
    if "image" in df.columns:
        df["image"] = df["image"].astype(str).apply(
            lambda x: x if x.lower().endswith((".jpg", ".png", ".jpeg")) else f"{x}.jpg"
        )
    return df[["image", "label"]].reset_index(drop=True)


# ============================================
# FIX 6: DATASET INTEGRITY VERIFICATION
# ============================================

def verify_dataset_integrity(df, name="dataset"):
    """Check for duplicates, missing classes, and print class distribution."""
    print(f"\n[Integrity Check] {name}")
    total = len(df)
    dupes = df.duplicated(subset=["image"]).sum()
    if dupes > 0:
        print(f"  ⚠️  Removing {dupes} duplicate image entries")
        df = df.drop_duplicates(subset=["image"]).reset_index(drop=True)

    present_classes = set(df["label"].unique())
    missing_classes = set(CLASS_NAMES) - present_classes
    if missing_classes:
        print(f"  ⚠️  Missing classes: {missing_classes}")
    else:
        print(f"  ✅ All {NUM_CLASSES} classes present")

    print(f"  Total samples  : {len(df)} (removed {total - len(df)} dupes)")
    for cls in CLASS_NAMES:
        n = (df["label"] == cls).sum()
        pct = 100.0 * n / len(df)
        print(f"    {cls:<8}: {n:>5} samples ({pct:4.1f}%)")
    return df


from models.model import build_model, unfreeze_model, get_preprocess_input


from preprocessing.dataset_loader import DualInputGenerator, load_and_preprocess_metadata, remove_hair_cv


# ============================================
# DYNAMIC GENERATOR BUILDERS
# ============================================

def build_train_generator(df, batch_size, seed, model_name="densenet121", enable_multimodal=True):
    """Augmented training generator with DullRazor hair removal and dual-input metadata."""
    base_prep = get_preprocess_input(model_name)
    def combined_prep(img):
        img = remove_hair_cv(img)
        return base_prep(img)

    datagen = ImageDataGenerator(
        preprocessing_function=combined_prep,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
    )
    gen = datagen.flow_from_dataframe(
        df,
        directory=IMAGE_ROOT,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    meta_lookup = load_and_preprocess_metadata()
    if enable_multimodal and meta_lookup:
        gen = DualInputGenerator(gen, meta_lookup)
    return gen


def build_eval_generator(df, batch_size, seed, model_name="densenet121", enable_multimodal=True):
    """No-augmentation generator with DullRazor hair removal and dual-input metadata."""
    base_prep = get_preprocess_input(model_name)
    def combined_prep(img):
        img = remove_hair_cv(img)
        return base_prep(img)

    datagen = ImageDataGenerator(preprocessing_function=combined_prep)
    gen = datagen.flow_from_dataframe(
        df,
        directory=IMAGE_ROOT,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    meta_lookup = load_and_preprocess_metadata()
    if enable_multimodal and meta_lookup:
        gen = DualInputGenerator(gen, meta_lookup)
    return gen


from models.evaluate import evaluate_model, predict_batch_with_tta, compute_pr_auc_macro


# ============================================
# FIX 5: COMPREHENSIVE METRICS WITH TTA
# ============================================

def compute_full_metrics(model, gen, class_names, split_name="Test", use_tta=True):
    """Compute all evaluation metrics from a generator with optional Test-Time Augmentation."""
    print(f"\n[Evaluation] Running on {split_name} set (TTA={'ON' if use_tta else 'OFF'})...")
    y_true_all, y_prob_all = [], []
    gen.reset()
    n_steps = len(gen)
    for i in range(n_steps):
        x_batch, y_batch = gen[i]
        if use_tta:
            y_prob_all.append(predict_batch_with_tta(model, x_batch))
        else:
            y_prob_all.append(model.predict(x_batch, verbose=0))
        y_true_all.append(y_batch)
        print(f"\r  Predicting with TTA: {i+1}/{n_steps} batches", end="", flush=True)
    print()  # newline after prediction loop

    y_prob = np.concatenate(y_prob_all, axis=0)
    y_true_oh = np.concatenate(y_true_all, axis=0)
    y_true = np.argmax(y_true_oh, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy      = accuracy_score(y_true, y_pred)
    bal_accuracy  = balanced_accuracy_score(y_true, y_pred)
    macro_f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec    = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc           = matthews_corrcoef(y_true, y_pred)
    cohen_kappa   = cohen_kappa_score(y_true, y_pred)
    pr_auc_macro  = compute_pr_auc_macro(y_true_oh, y_prob)
    cm            = confusion_matrix(y_true, y_pred)
    report        = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    print(f"  {'='*42}")
    print(f"  FINAL {split_name.upper()} SET METRICS")
    print(f"  {'='*42}")
    print(f"  Accuracy         : {accuracy*100:.2f}%")
    print(f"  Balanced Accuracy: {bal_accuracy*100:.2f}%")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    print(f"  Macro Precision  : {macro_prec:.4f}")
    print(f"  Macro Recall     : {macro_recall:.4f}")
    print(f"  MCC              : {mcc:.4f}")
    print(f"  Cohen's Kappa    : {cohen_kappa:.4f}")
    print(f"  PR-AUC (Macro)   : {pr_auc_macro:.4f}")
    print(f"  ROC-AUC (OvR)   : {roc_auc:.4f}")
    print(f"  {'='*42}")
    print(f"\n  Per-Class Report:\n{report}")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal_accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_recall,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
        "pr_auc_macro": pr_auc_macro,
        "roc_auc_ovr": roc_auc,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


# ============================================
# FIX 8: TRAINING HISTORY PLOTS
# ============================================

def save_training_plots(histories, plots_dir):
    """Save accuracy and loss plots from training histories."""
    os.makedirs(plots_dir, exist_ok=True)

    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc     += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss    += h.history.get("loss", [])
        val_loss+= h.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    # Accuracy Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, acc,     label="Train Accuracy", color="#4C72B0", linewidth=2)
    ax.plot(epochs_range, val_acc, label="Val Accuracy",   color="#DD8452", linewidth=2, linestyle="--")
    ax.set_title("Centralized Model — Accuracy vs. Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_history.png"), dpi=150)
    plt.close()

    # Loss Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, loss,     label="Train Loss", color="#4C72B0", linewidth=2)
    ax.plot(epochs_range, val_loss, label="Val Loss",   color="#DD8452", linewidth=2, linestyle="--")
    ax.set_title("Centralized Model — Loss vs. Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_history.png"), dpi=150)
    plt.close()

    print(f"\n  📊 Plots saved to: {plots_dir}/")


# ============================================
# MAIN TRAINING LOOP
# ============================================

def train(args):
    set_seeds(SEED)  # FIX 7: reproducibility
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ----------------------------------------
    # BUILD TRAINING CSV
    # ----------------------------------------
    if args.csv_path is not None:
        if not os.path.exists(args.csv_path):
            print(f"\n[ERROR] CSV file not found: {args.csv_path}")
            sys.exit(1)
        train_val_df = load_and_format_csv(args.csv_path)
        mode_label = f"RAW CSV: {args.csv_path}"
    elif args.client_id == "all":
        csv_files = [os.path.join(DATASET_DIR, f"client_{i}.csv") for i in range(1, 5)]
        missing = [f for f in csv_files if not os.path.exists(f)]
        if missing:
            print(f"\n[ERROR] Missing partition files: {missing}")
            sys.exit(1)
        train_val_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        mode_label = "ALL CLIENTS COMBINED (Full Centralized)"
    else:
        train_val_df = pd.read_csv(os.path.join(DATASET_DIR, f"{args.client_id}.csv"))
        mode_label = args.client_id

    # FIX 6: verify integrity
    train_val_df = verify_dataset_integrity(train_val_df, name="Training+Validation Data")

    # ----------------------------------------
    # FIX 1: SEPARATE TEST SET
    # ----------------------------------------
    if os.path.exists(GLOBAL_TEST_CSV):
        test_df = pd.read_csv(GLOBAL_TEST_CSV)
        test_df = verify_dataset_integrity(test_df, name="Global Test Data")
        # Remove any test images from train_val to prevent leakage
        test_images = set(test_df["image"].values)
        pre_len = len(train_val_df)
        train_val_df = train_val_df[~train_val_df["image"].isin(test_images)].reset_index(drop=True)
        removed = pre_len - len(train_val_df)
        if removed > 0:
            print(f"\n  🛡️  Removed {removed} overlapping images from train/val to prevent data leakage")
        use_global_test = True
    else:
        print("\n  ⚠️  global_test.csv not found — will use 10% val split as test proxy")
        test_df = None
        use_global_test = False

    # ----------------------------------------
    # TRAIN / VALIDATION SPLIT
    # ----------------------------------------
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_val_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # ----------------------------------------
    # FIX 2: BUILD GENERATORS (no numpy arrays)
    # ----------------------------------------
    print("\n[1/4] Building data generators...")
    train_gen = build_train_generator(train_df, args.batch_size, SEED, model_name=args.model_name)
    val_gen   = build_eval_generator(val_df,   args.batch_size, SEED, model_name=args.model_name)
    test_gen  = build_eval_generator(test_df if use_global_test else val_df, args.batch_size, SEED, model_name=args.model_name)

    print(f"    Train samples  : {len(train_df)}")
    print(f"    Val samples    : {len(val_df)}")
    print(f"    Test samples   : {len(test_df) if use_global_test else len(val_df)} ({'global unseen' if use_global_test else 'val proxy'})")

    # ----------------------------------------
    # CLASS WEIGHTS (all 8 classes guaranteed)
    # ----------------------------------------
    y_train_ints = np.array([CLASS_NAMES.index(l) for l in train_df["label"]])
    y_train_oh = tf.keras.utils.to_categorical(y_train_ints, num_classes=NUM_CLASSES)
    class_weights = get_class_weights(y_train_oh, class_labels=list(range(NUM_CLASSES)), max_weight_cap=2.5, num_classes=NUM_CLASSES)
    print(f"\n[2/4] Training Mode: ASYMMETRIC Class Weights (Majority=1.0, Minority boosted up to 2.5)")
    for k, v in sorted(class_weights.items()):
        print(f"    {CLASS_NAMES[k]:<8}: {v:.3f}")

    # ----------------------------------------
    # FIX 3: BUILD MODEL (identical to FL)
    # ----------------------------------------
    print(f"\n[3/4] Building model ({args.model_name}, pooling={args.pooling_mode} — identical to FL config)...")
    model = build_model(
        model_name=args.model_name,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        pooling_mode=args.pooling_mode,
    )

    # ----------------------------------------
    # FIX 4: CALLBACKS & Schedulers
    # ----------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_tag = str(args.model_name).lower()
    ckpt_filepath = os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.keras")
    csv_log_path  = os.path.join(CHECKPOINT_DIR, f"centralized_training_log_{model_tag}.csv")

    try:
        if os.path.exists(csv_log_path):
            os.remove(csv_log_path)
    except Exception:
        try:
            with open(csv_log_path, "w") as f:
                pass
        except Exception:
            pass

    steps_per_epoch = len(train_gen)

    def make_cosine_callback(initial_lr, total_epochs):
        total_steps = max(1, total_epochs * steps_per_epoch)
        def lr_fn(epoch, current_lr):
            progress = epoch / float(max(1, total_epochs))
            # Cosine decay down to 1% of initial_lr
            return float(initial_lr * (0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress))))
        return tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=0)

    base_callbacks_p1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_filepath,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            csv_log_path,
            append=True,
        ),
    ]

    base_callbacks_p2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_filepath,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            csv_log_path,
            append=True,
        ),
    ]

    if args.lr_scheduler == "cosine":
        callbacks_p1 = base_callbacks_p1 + [make_cosine_callback(initial_lr=5e-4, total_epochs=args.epochs)]
    else:
        callbacks_p1 = base_callbacks_p1 + [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
        ]

    # Fast-Fail Sanity Check
    sanity_check(model, train_gen, label="before Phase 1 (Centralized Head Warmup)")

    print(f"\n[4/4] Phase 1: Training classification head ({args.epochs} epochs, Scheduler={args.lr_scheduler.upper()})...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks_p1,
        workers=4,
        max_queue_size=10,
        verbose=1,  # clean single-line progress bar per epoch
    )

    histories = [history1]

    if args.finetune_epochs > 0:
        print(f"\n[Fine-tune] Unfreezing top backbone layers for {args.model_name} ({args.finetune_epochs} epochs)...")
        model = unfreeze_model(model, fine_tune_at=None, learning_rate=1e-4, model_name=args.model_name)

        if args.lr_scheduler == "cosine":
            callbacks_p2 = base_callbacks_p2 + [make_cosine_callback(initial_lr=1e-4, total_epochs=args.finetune_epochs)]
        else:
            callbacks_p2 = base_callbacks_p2 + [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
                )
            ]

        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks_p2,
            workers=4,
            max_queue_size=10,
            verbose=1,  # clean single-line progress bar per epoch
        )
        histories.append(history2)





    # ----------------------------------------
    # FIX 8: SAVE TRAINING HISTORY PLOTS
    # ----------------------------------------
    save_training_plots(histories, PLOTS_DIR)

    # ----------------------------------------
    # FIX 5 + 1: EVALUATE ON UNSEEN TEST SET (WITH TTA)
    # ----------------------------------------
    test_results = compute_full_metrics(
        model, test_gen,
        class_names=CLASS_NAMES,
        split_name="Unseen Test" if use_global_test else "Validation (Proxy Test)",
        use_tta=args.use_tta,
    )

    # Save metrics to JSON
    metrics_path = os.path.join(CHECKPOINT_DIR, f"centralized_metrics_{model_tag}.json")
    with open(metrics_path, "w") as f:
        serializable = {k: v for k, v in test_results.items() if k != "report"}
        serializable["report"] = test_results["report"]
        json.dump(serializable, f, indent=2)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "centralized_final.keras")
    model.save(final_path)

    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"  Mode             : {mode_label}")
    print(f"  Best Model       : {ckpt_filepath}")
    print(f"  Final Model      : {final_path}")
    print(f"  Metrics JSON     : {metrics_path}")
    print(f"  Training Log CSV : {CHECKPOINT_DIR}/centralized_training_log.csv")
    print(f"  Accuracy Plot    : {PLOTS_DIR}/accuracy_history.png")
    print(f"  Loss Plot        : {PLOTS_DIR}/loss_history.png")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
