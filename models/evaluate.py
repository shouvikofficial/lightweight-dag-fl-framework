from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_model(
    model,
    x_test,
    y_test,
    class_names=None,
    batch_size=32,
    top_k: Optional[int] = 3,
):
    """
    Evaluate a Keras model on test data.

    Parameters
    ----------
    model       : compiled tf.keras.Model
    x_test      : numpy array of test images
    y_test      : one-hot encoded test labels
    class_names : optional list of class name strings
    batch_size  : int

    Returns
    -------
    dict containing loss, accuracy, macro_f1,
    classification report, and confusion matrix
    """

    # ========================================
    # KERAS EVALUATION
    # ========================================

    eval_results = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=0
    )

    if isinstance(eval_results, (list, tuple)):
        avg_loss, accuracy = eval_results[0], eval_results[1]
    else:
        avg_loss, accuracy = float(eval_results), 0.0

    # ========================================
    # PREDICTIONS
    # ========================================

    y_pred_prob = model.predict(
        x_test,
        batch_size=batch_size,
        verbose=0
    )

    # Convert one-hot → integer labels
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ========================================
    # METRICS
    # ========================================

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm
    }

    if top_k is not None:
        top_k_acc = tf.keras.metrics.top_k_categorical_accuracy(
            y_test,
            y_pred_prob,
            k=top_k,
        )
        results["top_k_accuracy"] = float(np.mean(top_k_acc))

    return results


# ============================================
# EVALUATE WITH GENERATOR
# ============================================

def evaluate_with_generator(
    model,
    val_generator,
    steps=None,
    class_names=None
):
    """
    Evaluate using a Keras ImageDataGenerator.

    Parameters
    ----------
    model         : compiled tf.keras.Model
    val_generator : Keras data generator
    steps         : int or None (auto-computed)
    class_names   : optional list of class name strings

    Returns
    -------
    dict containing loss and accuracy
    """

    results = model.evaluate(
        val_generator,
        steps=steps,
        verbose=0
    )

    return {
        "loss": results[0],
        "accuracy": results[1]
    }