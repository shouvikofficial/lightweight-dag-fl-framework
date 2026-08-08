"""
===================================================
 DUAL-MODEL ENSEMBLE EVALUATION (TTA = ON)
===================================================

Loads trained model checkpoints (DenseNet201 + EfficientNetB0),
runs 5-pass Test-Time Augmentation (TTA) on unseen global test set,
averages prediction probabilities, and computes ensemble metrics.

Usage:
    python eval_ensemble.py --model1 densenet201 --model2 efficientnetb0
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.model import build_model, GeMPooling2D, CBAM, CategoricalFocalLoss
from models.evaluate import compute_pr_auc_macro, predict_batch_with_tta
from preprocessing.dataset_loader import prepare_global_test_generator, CLASS_NAMES, IMAGE_SIZE

GLOBAL_TEST_CSV   = "dataset/partitions/global_test.csv"
IMAGE_ROOT        = "dataset/raw/ISIC_2019_Training_Input"
CHECKPOINT_DIR    = "models/checkpoints"
OUTPUT_JSON       = "models/checkpoints/ensemble_metrics.json"


# ─────────────────────────────────────────────
#  CUSTOM OBJECT MAP
# ─────────────────────────────────────────────
CUSTOM_OBJECTS = {
    "GeMPooling2D": GeMPooling2D,
    "CBAM": CBAM,
    "CategoricalFocalLoss": CategoricalFocalLoss,
}


def load_trained_model(model_name: str):
    """
    Tries to load a trained model in this priority order:
      1. centralized_full_{name}/   — full SavedModel directory (best, no issues)
      2. centralized_best_{name}.keras — full model saved by ModelCheckpoint
      3. centralized_best_{name}.h5    — full model in HDF5 format
    """
    model_tag = model_name.lower()

    candidates = [
        os.path.join(CHECKPOINT_DIR, f"centralized_full_{model_tag}"),       # full SavedModel dir
        os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.keras"),  # full .keras
        os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.h5"),     # full .h5
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        print(f"  Trying: {path}")
        try:
            model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
            print(f"  ✅ Loaded full model for {model_name.upper()} from: {path}")
            return model
        except Exception as e:
            print(f"  ⚠️  load_model failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        f"\n❌ CANNOT LOAD MODEL: {model_name}\n"
        f"   None of the full-model checkpoints could be loaded.\n"
        f"   Root cause: the existing checkpoints were saved with save_weights_only=True\n"
        f"   (weights-only files, not full models). Keras 2.10 cannot reload these\n"
        f"   for custom-layer models without triggering an 'axes don't match array' error.\n\n"
        f"   ✅ FIX (already applied to train_local.py):\n"
        f"      Retrain each model once — the new code saves the full model:\n\n"
        f"      python train_local.py --model_name densenet201   --epochs 10 --finetune_epochs 25\n"
        f"      python train_local.py --model_name efficientnetb0 --epochs 10 --finetune_epochs 25\n\n"
        f"   After retraining, run this script again. Ensemble will work."
    )


# ─────────────────────────────────────────────
#  TTA RUNNER
# ─────────────────────────────────────────────

def run_tta_on_generator(model, gen):
    """Runs 5-view TTA across all batches. Returns (y_true_onehot, y_probs)."""
    gen.reset()
    n_batches = len(gen)
    y_prob_list, y_true_list = [], []

    for i in range(n_batches):
        x_batch, y_batch = gen[i]
        probs = predict_batch_with_tta(model, x_batch)
        y_prob_list.append(probs)
        y_true_list.append(y_batch)
        print(f"\r  Evaluating TTA batch {i + 1}/{n_batches}", end="", flush=True)

    print()
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_prob_list, axis=0)


# ─────────────────────────────────────────────
#  MAIN ENSEMBLE EVALUATION
# ─────────────────────────────────────────────

def evaluate_ensemble(model1_name="densenet201", model2_name="efficientnetb0"):
    print("=" * 60)
    print(f" 🤖 DUAL-MODEL ENSEMBLE EVALUATION ({model1_name.upper()} + {model2_name.upper()})")
    print("=" * 60)

    # [1/4] Test generators
    print("\n[1/4] Preparing test generators...")
    test_gen_m1 = prepare_global_test_generator(
        csv_path=GLOBAL_TEST_CSV,
        image_root=IMAGE_ROOT,
        batch_size=16,
        model_name=model1_name,
        enable_multimodal=True,
    )
    test_gen_m2 = prepare_global_test_generator(
        csv_path=GLOBAL_TEST_CSV,
        image_root=IMAGE_ROOT,
        batch_size=16,
        model_name=model2_name,
        enable_multimodal=True,
    )

    # [2/4] Load models
    print("\n[2/4] Building models and loading weights...")
    m1 = load_trained_model(model1_name)
    m2 = load_trained_model(model2_name)

    # [3/4] TTA Inference
    print("\n[3/4] Running TTA Evaluation on Unseen Test Set (5 views per model)...")
    y_true_oh, probs_m1 = run_tta_on_generator(m1, test_gen_m1)
    _,         probs_m2 = run_tta_on_generator(m2, test_gen_m2)
    print("✅ TTA Inference Complete!")

    # [4/4] Ensemble
    print("\n[4/4] Computing Dual-Model Ensemble Consensus Probabilities...")
    ensemble_probs = 0.5 * probs_m1 + 0.5 * probs_m2
    y_true = np.argmax(y_true_oh, axis=1)
    y_pred = np.argmax(ensemble_probs, axis=1)

    acc       = float(accuracy_score(y_true, y_pred))
    bal_acc   = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec= float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    kappa     = float(cohen_kappa_score(y_true, y_pred))
    mcc       = float(matthews_corrcoef(y_true, y_pred))
    pr_auc    = float(compute_pr_auc_macro(y_true_oh, ensemble_probs))

    try:
        roc_auc = float(roc_auc_score(y_true_oh, ensemble_probs, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = 0.0

    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4)

    print("\n" + "=" * 50)
    print(" 🏆 FINAL DUAL-MODEL ENSEMBLE METRICS (TTA=ON)")
    print("=" * 50)
    print(f"  Models Ensembled : {model1_name.upper()} + {model2_name.upper()}")
    print(f"  Top-1 Accuracy   : {acc * 100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"  ROC-AUC (OvR)    : {roc_auc:.4f}")
    print(f"  PR-AUC (Macro)   : {pr_auc:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    print(f"  Macro Precision  : {macro_prec:.4f}")
    print(f"  Macro Recall     : {macro_rec:.4f}")
    print(f"  MCC              : {mcc:.4f}")
    print(f"  Cohen's Kappa    : {kappa:.4f}")
    print("=" * 50)
    print("\nDetailed Per-Class Classification Report:\n")
    print(report_str)

    metrics_payload = {
        "model1":              model1_name,
        "model2":              model2_name,
        "top1_accuracy":       acc,
        "balanced_accuracy":   bal_acc,
        "macro_f1":            macro_f1,
        "macro_precision":     macro_prec,
        "macro_recall":        macro_rec,
        "cohen_kappa":         kappa,
        "mcc":                 mcc,
        "roc_auc_ovr":         roc_auc,
        "pr_auc_macro":        pr_auc,
        "classification_report": report_str,
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n✅ Saved Ensemble Metrics to: {OUTPUT_JSON}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="densenet201")
    parser.add_argument("--model2", type=str, default="efficientnetb0")
    args = parser.parse_args()
    evaluate_ensemble(args.model1, args.model2)


if __name__ == "__main__":
    main()
