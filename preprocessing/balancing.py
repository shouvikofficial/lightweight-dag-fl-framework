from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(
    labels: np.ndarray,
    class_labels: Optional[Iterable[int]] = None,
    num_classes: int = 8,
) -> Dict[int, float]:
    """Compute balanced class weights for integer or one-hot labels.

    Ensures all expected class IDs (0..num_classes-1) receive a valid float weight,
    even if rare classes are absent from a local batch sample.
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
        num_classes = max(num_classes, y.shape[1])
    else:
        raise ValueError("labels must be 1D or 2D array")

    if class_labels is None:
        target_classes = np.arange(num_classes)
    else:
        target_classes = np.asarray(list(class_labels), dtype=int)
        if target_classes.size == 0:
            raise ValueError("class_labels must not be empty")

    n_samples = len(y_int)
    n_classes = len(target_classes)

    counts = np.bincount(y_int, minlength=max(target_classes) + 1)

    weights = {}
    max_observed_w = 1.0
    for c in target_classes:
        cnt = counts[c] if c < len(counts) else 0
        if cnt > 0:
            w = n_samples / (n_classes * float(cnt))
            weights[int(c)] = float(w)
            if w > max_observed_w:
                max_observed_w = w
        else:
            weights[int(c)] = -1.0

    # Fill zero-count classes with the max observed weight cap
    for c in target_classes:
        if weights[int(c)] < 0:
            weights[int(c)] = float(max_observed_w)

    return weights