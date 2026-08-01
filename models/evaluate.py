from typing import Dict, List, Optional

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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)


# ============================================
# SOTA IMPROVEMENT 5: TEST-TIME AUGMENTATION (TTA)
# ============================================

def predict_batch_with_tta(model, inputs) -> np.ndarray:
    """
    Predict probabilities across 5 spatial variations of image batch (with metadata passed through):
      1. Original
      2. Horizontal Flip
      3. Vertical Flip
      4. Horizontal + Vertical Flip
      5. 90-degree Rotation
    Averages predictions across all 5 views.
    """
    if isinstance(inputs, (list, tuple)):
        x_img, x_meta = inputs[0], inputs[1]
        p1 = model.predict([x_img, x_meta], verbose=0)
        p2 = model.predict([np.flip(x_img, axis=2), x_meta], verbose=0)
        p3 = model.predict([np.flip(x_img, axis=1), x_meta], verbose=0)
        p4 = model.predict([np.flip(np.flip(x_img, axis=1), axis=2), x_meta], verbose=0)
        p5 = model.predict([np.rot90(x_img, k=1, axes=(1, 2)), x_meta], verbose=0)
        return (p1 + p2 + p3 + p4 + p5) / 5.0
    else:
        x_img = inputs
        p1 = model.predict(x_img, verbose=0)
        p2 = model.predict(np.flip(x_img, axis=2), verbose=0)
        p3 = model.predict(np.flip(x_img, axis=1), verbose=0)
        p4 = model.predict(np.flip(np.flip(x_img, axis=1), axis=2), verbose=0)
        p5 = model.predict(np.rot90(x_img, k=1, axes=(1, 2)), verbose=0)
        return (p1 + p2 + p3 + p4 + p5) / 5.0


def compute_pr_auc_macro(y_true_oh: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Macro Precision-Recall AUC across all classes."""
    pr_aucs = []
    n_classes = y_true_oh.shape[1]
    for c in range(n_classes):
        if np.sum(y_true_oh[:, c]) > 0:
            p, r, _ = precision_recall_curve(y_true_oh[:, c], y_prob[:, c])
            pr_aucs.append(auc(r, p))
    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


# ============================================
# EVALUATE ON NUMPY ARRAYS
# ============================================

def evaluate_model(
    model,
    x_test,
    y_test,
    class_names=None,
    batch_size=32,
    top_k: Optional[int] = 3,
    use_tta: bool = True,
) -> Dict:
    eval_results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    avg_loss = float(eval_results[0]) if isinstance(eval_results, (list, tuple)) else float(eval_results)

    if use_tta:
        y_pred_prob = predict_batch_with_tta(model, x_test)
    else:
        y_pred_prob = model.predict(x_test, batch_size=batch_size, verbose=0)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc          = matthews_corrcoef(y_true, y_pred)
    cohen_kappa  = cohen_kappa_score(y_true, y_pred)
    pr_auc_macro = compute_pr_auc_macro(y_test, y_pred_prob)
    cm           = confusion_matrix(y_true, y_pred)
    report       = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    results = {
        "loss":             avg_loss,
        "accuracy":         accuracy,
        "balanced_accuracy": bal_accuracy,
        "macro_f1":         macro_f1,
        "macro_precision":  macro_prec,
        "macro_recall":     macro_recall,
        "mcc":              mcc,
        "cohen_kappa":      cohen_kappa,
        "pr_auc_macro":     pr_auc_macro,
        "roc_auc_ovr":      roc_auc,
        "confusion_matrix": cm,
        "report":           report,
    }

    if top_k is not None:
        topk = tf.keras.metrics.top_k_categorical_accuracy(y_test, y_pred_prob, k=top_k)
        results["top_k_accuracy"] = float(np.mean(topk))

    return results


# ============================================
# EVALUATE ON GENERATOR
# ============================================

def evaluate_with_generator(
    model,
    generator,
    class_names: Optional[List[str]] = None,
    steps: Optional[int] = None,
    use_tta: bool = True,
) -> Dict:
    generator.reset()
    if steps is None:
        steps = len(generator)

    y_true_all, y_prob_all = [], []
    for i in range(steps):
        x_batch, y_batch = generator[i]
        if use_tta:
            y_prob_all.append(predict_batch_with_tta(model, x_batch))
        else:
            y_prob_all.append(model.predict(x_batch, verbose=0))
        y_true_all.append(y_batch)

    y_prob   = np.concatenate(y_prob_all, axis=0)
    y_true_oh = np.concatenate(y_true_all, axis=0)
    y_true   = np.argmax(y_true_oh, axis=1)
    y_pred   = np.argmax(y_prob, axis=1)

    eval_results = model.evaluate(generator, steps=steps, verbose=0)
    avg_loss = float(eval_results[0]) if isinstance(eval_results, (list, tuple)) else float(eval_results)

    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc          = matthews_corrcoef(y_true, y_pred)
    cohen_kappa  = cohen_kappa_score(y_true, y_pred)
    pr_auc_macro = compute_pr_auc_macro(y_true_oh, y_prob)
    cm           = confusion_matrix(y_true, y_pred)
    report       = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    return {
        "loss":             avg_loss,
        "accuracy":         accuracy,
        "balanced_accuracy": bal_accuracy,
        "macro_f1":         macro_f1,
        "macro_precision":  macro_prec,
        "macro_recall":     macro_recall,
        "mcc":              mcc,
        "cohen_kappa":      cohen_kappa,
        "pr_auc_macro":     pr_auc_macro,
        "roc_auc_ovr":      roc_auc,
        "confusion_matrix": cm,
        "report":           report,
    }