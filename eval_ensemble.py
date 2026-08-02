"""
===================================================
 DUAL-MODEL ENSEMBLE EVALUATION (TTA = ON)
===================================================

Loads trained checkpoints:
  1. DenseNet201 (centralized_best_densenet201.keras)
  2. ResNet50V2 (centralized_best_resnet50v2.keras)

Evaluates averaged predictions across 5 TTA views on the unseen global test set.
Saves comprehensive metrics to models/checkpoints/ensemble_metrics.json.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.evaluate import predict_batch_with_tta, compute_pr_auc_macro
from preprocessing.dataset_loader import prepare_global_test_generator, CLASS_NAMES

# ============================================
# CONFIGURATION
# ============================================

GLOBAL_TEST_CSV = "dataset/partitions/global_test.csv"
IMAGE_ROOT = "dataset/raw/ISIC_2019_Training_Input"

DENSENET_PATH = "models/checkpoints/centralized_best_densenet201.keras"
RESNET_PATH = "models/checkpoints/centralized_best_resnet50v2.keras"
OUTPUT_JSON = "models/checkpoints/ensemble_metrics.json"


def evaluate_ensemble():
    print("=" * 60)
    print(" 🤖 DUAL-MODEL ENSEMBLE EVALUATION (DenseNet201 + ResNet50V2)")
    print("=" * 60)

    # 1. Verify Checkpoint Existence
    if not os.path.exists(DENSENET_PATH):
        raise FileNotFoundError(f"Missing DenseNet201 checkpoint: {DENSENET_PATH}")
    if not os.path.exists(RESNET_PATH):
        raise FileNotFoundError(f"Missing ResNet50V2 checkpoint: {RESNET_PATH}")

    # 2. Load Generator for DenseNet201
    print("\n[1/4] Preparing test generators...")
    test_gen_dense = prepare_global_test_generator(
        GLOBAL_TEST_CSV, IMAGE_ROOT, model_name="densenet201"
    )
    
    # Load Generator for ResNet50V2
    test_gen_resnet = prepare_global_test_generator(
        GLOBAL_TEST_CSV, IMAGE_ROOT, model_name="resnet50v2"
    )

    # 3. Load Trained Models
    print(f"\n[2/4] Loading DenseNet201 model from {DENSENET_PATH}...")
    model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)

    print(f"[2/4] Loading ResNet50V2 model from {RESNET_PATH}...")
    model_resnet = tf.keras.models.load_model(RESNET_PATH, compile=False)

    # 4. Predict with TTA for Both Models
    print("\n[3/4] Running TTA Evaluation on Unseen Test Set (5 Spatial Views)...")
    y_prob_dense = []
    y_prob_resnet = []
    y_true_all = []

    n_steps = len(test_gen_dense)
    for i in range(n_steps):
        x_dense, y_batch = test_gen_dense[i]
        x_resnet, _ = test_gen_resnet[i]

        p_dense = predict_batch_with_tta(model_dense, x_dense)
        p_resnet = predict_batch_with_tta(model_resnet, x_resnet)

        y_prob_dense.append(p_dense)
        y_prob_resnet.append(p_resnet)
        y_true_all.append(y_batch)

        print(f"\r  Evaluating TTA batch {i+1}/{n_steps}", end="", flush=True)

    print("\n✅ TTA Inference Complete!")

    y_prob_dense = np.concatenate(y_prob_dense, axis=0)
    y_prob_resnet = np.concatenate(y_prob_resnet, axis=0)
    y_true_oh = np.concatenate(y_true_all, axis=0)

    # 5. Dual-Model Ensemble Probability Averaging
    print("\n[4/4] Computing Dual-Model Ensemble Consensus Probabilities...")
    y_prob_ensemble = (y_prob_dense + y_prob_resnet) / 2.0

    y_true = np.argmax(y_true_oh, axis=1)
    y_pred = np.argmax(y_prob_ensemble, axis=1)

    # 6. Compute Full Medical Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    pr_auc = compute_pr_auc_macro(y_true_oh, y_prob_ensemble)

    try:
        roc_auc = roc_auc_score(y_true_oh, y_prob_ensemble, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    print("\n" + "=" * 50)
    print(" 🏆 FINAL DUAL-MODEL ENSEMBLE METRICS (TTA=ON)")
    print("=" * 50)
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
    print(report)

    # 7. Save Metrics JSON
    results = {
        "model_type": "Dual-Model Ensemble (DenseNet201 + ResNet50V2)",
        "tta_enabled": True,
        "test_samples": len(y_true),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "roc_auc_ovr": float(roc_auc),
        "pr_auc_macro": float(pr_auc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved Ensemble Metrics to: {OUTPUT_JSON}")


if __name__ == "__main__":
    evaluate_ensemble()
