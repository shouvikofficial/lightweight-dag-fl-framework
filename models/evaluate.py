from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
) -> Dict:
    """
    Evaluate a Keras model on pre-loaded numpy test arrays.

    Returns a dict containing:
        loss, accuracy, balanced_accuracy, macro_f1,
        macro_precision, macro_recall, roc_auc_ovr,
        confusion_matrix, report, top_k_accuracy (optional).
    """

    eval_results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    if isinstance(eval_results, (list, tuple)):
        avg_loss = float(eval_results[0])
    else:
        avg_loss = float(eval_results)

    y_pred_prob = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
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
) -> Dict:
    """
    Evaluate using a Keras ImageDataGenerator (memory-safe).

    Returns a dict containing:
        loss, accuracy, balanced_accuracy, macro_f1,
        macro_precision, macro_recall, roc_auc_ovr,
        confusion_matrix, report.
    """

    generator.reset()
    if steps is None:
        steps = len(generator)

    y_true_all, y_prob_all = [], []
    for i in range(steps):
        x_batch, y_batch = generator[i]
        y_prob_all.append(model.predict(x_batch, verbose=0))
        y_true_all.append(y_batch)

    y_prob   = np.concatenate(y_prob_all, axis=0)
    y_true_oh = np.concatenate(y_true_all, axis=0)
    y_true   = np.argmax(y_true_oh, axis=1)
    y_pred   = np.argmax(y_prob, axis=1)

    # Keras native loss/accuracy
    eval_results = model.evaluate(generator, steps=steps, verbose=0)
    avg_loss = float(eval_results[0]) if isinstance(eval_results, (list, tuple)) else float(eval_results)

    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
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
        "roc_auc_ovr":      roc_auc,
        "confusion_matrix": cm,
        "report":           report,
    }