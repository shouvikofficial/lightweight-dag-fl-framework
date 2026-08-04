"""
===================================================
 DUAL-MODEL ENSEMBLE EVALUATION (TTA = ON)
===================================================

Loads trained model checkpoints (e.g. DenseNet201 + EfficientNetB0),
runs 5-pass Test-Time Augmentation (TTA) for both models on unseen global test set,
averages prediction probabilities, and computes complete ensemble metrics.

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
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.model import build_model, GeMPooling2D, CBAM, CategoricalFocalLoss
from models.evaluate import predict_batch_with_tta, compute_pr_auc_macro
from preprocessing.dataset_loader import prepare_global_test_generator, CLASS_NAMES, IMAGE_SIZE

GLOBAL_TEST_CSV = "dataset/partitions/global_test.csv"
IMAGE_ROOT = "dataset/raw/ISIC_2019_Training_Input"
CHECKPOINT_DIR = "models/checkpoints"
OUTPUT_JSON = "models/checkpoints/ensemble_metrics.json"


import zipfile

def load_trained_model(model_name: str):
    """Builds model structure and loads best saved weights or full model."""
    model_tag = str(model_name).lower()
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.keras")
    
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.h5")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {ckpt_path}")

    print(f"Loading checkpoint for {model_name} from: {ckpt_path}")
    custom_objs = {
        "GeMPooling2D": GeMPooling2D,
        "CBAM": CBAM,
        "CategoricalFocalLoss": CategoricalFocalLoss,
    }
    
    # 1. Attempt full model load
    try:
        model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objs, compile=False)
        return model
    except Exception:
        pass

    # 2. Build model architecture
    model = build_model(
        model_name=model_name,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=len(CLASS_NAMES),
        pooling_mode="gem_gap",
    )
    
    # 3. Handle .keras Zip Archive Format
    if zipfile.is_zipfile(ckpt_path):
        tmp_dir = os.path.join(CHECKPOINT_DIR, f"tmp_{model_tag}")
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(ckpt_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        h5_weights = None
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".h5"):
                    h5_weights = os.path.join(root, f)
                    break
        if h5_weights:
            model.load_weights(h5_weights, by_name=True, skip_mismatch=True)
            return model

    # 4. Standard HDF5 load
    model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
    return model


def evaluate_ensemble(model1_name="densenet201", model2_name="efficientnetb0"):
    print("=" * 60)
    print(f" 🤖 DUAL-MODEL ENSEMBLE EVALUATION ({model1_name.upper()} + {model2_name.upper()})")
    print("=" * 60)

    print("\n[1/4] Preparing test generators...")
    test_gen1 = prepare_global_test_generator(
        GLOBAL_TEST_CSV, IMAGE_ROOT, model_name=model1_name
    )
    test_gen2 = prepare_global_test_generator(
        GLOBAL_TEST_CSV, IMAGE_ROOT, model_name=model2_name
    )

    print("\n[2/4] Building models and loading weights...")
    m1 = load_trained_model(model1_name)
    m2 = load_trained_model(model2_name)

    print("\n[3/4] Running TTA Evaluation on Unseen Test Set (5 Spatial Views per model)...")
    y_prob1 = []
    y_prob2 = []
    y_true_all = []

    n_steps = len(test_gen1)
    for i in range(n_steps):
        x1, y_batch = test_gen1[i]
        x2, _ = test_gen2[i]

        p1 = predict_batch_with_tta(m1, x1)
        p2 = predict_batch_with_tta(m2, x2)

        y_prob1.append(p1)
        y_prob2.append(p2)
        y_true_all.append(y_batch)

        print(f"\r  Evaluating TTA batch {i+1}/{n_steps}", end="", flush=True)

    print("\n✅ TTA Inference Complete!")

    y_prob1 = np.concatenate(y_prob1, axis=0)
    y_prob2 = np.concatenate(y_prob2, axis=0)
    y_true_oh = np.concatenate(y_true_all, axis=0)

    # 5. Dual-Model Ensemble Softmax Averaging
    print("\n[4/4] Computing Dual-Model Ensemble Consensus Probabilities...")
    y_prob_ensemble = (y_prob1 + y_prob2) / 2.0

    y_true = np.argmax(y_true_oh, axis=1)
    y_pred = np.argmax(y_prob_ensemble, axis=1)

    # 6. Compute Full Metrics
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
    print(report)

    # 7. Save Metrics JSON
    results = {
        "model_type": f"Dual-Model Ensemble ({model1_name.upper()} + {model2_name.upper()})",
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
        "report": report,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved Ensemble Metrics to: {OUTPUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Model Ensemble Evaluation")
    parser.add_argument("--model1", type=str, default="densenet201", help="First model backbone")
    parser.add_argument("--model2", type=str, default="efficientnetb0", help="Second model backbone")
    args = parser.parse_args()

    evaluate_ensemble(args.model1, args.model2)
