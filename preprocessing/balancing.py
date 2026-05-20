from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(
    labels: np.ndarray,
    class_labels: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    """Compute balanced class weights for integer or one-hot labels.

    Args:
        labels: Array of shape (n_samples,) with integer labels, or
            (n_samples, n_classes) with one-hot or probability-like labels.
        class_labels: Optional iterable of class ids to include. If omitted,
            classes are inferred from the data.

    Returns:
        Mapping of class id to weight.
    """

    if labels is None:
        raise ValueError("labels must not be None")

    y = np.asarray(labels)
    if y.size == 0:
        raise ValueError("labels must not be empty")

    if y.ndim == 1:
        y_int = y.astype(int)
    elif y.ndim == 2:
        y_int = np.argmax(y, axis=1).astype(int)
    else:
        raise ValueError("labels must be 1D or 2D array")

    if class_labels is None:
        classes = np.unique(y_int)
    else:
        classes = np.asarray(list(class_labels), dtype=int)
        if classes.size == 0:
            raise ValueError("class_labels must not be empty")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_int,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}