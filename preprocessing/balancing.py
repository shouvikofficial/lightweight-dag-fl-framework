from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(
    labels: np.ndarray,
    class_labels: Optional[Iterable[int]] = None,
    num_classes: int = 5,
    max_weight_cap: float = 1.8,
) -> Dict[int, float]:
    """Compute balanced class weights capped at max_weight_cap=1.8.

    Ensures all expected class IDs (0..num_classes-1) receive a valid float weight,
    while capping extreme weights at 3.0 to maximize top-1 diagnostic accuracy.
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
    for c in target_classes:
        cnt = counts[c] if c < len(counts) else 0
        if cnt > 0:
            w = n_samples / (n_classes * float(cnt))
            weights[int(c)] = float(w)
        else:
            weights[int(c)] = -1.0

    # -------------------------------------------------------------
    # ASYMMETRIC WEIGHT NORMALIZATION (Anchored at 1.0)
    # -------------------------------------------------------------
    # Instead of punishing majority classes with weights < 1.0,
    # we anchor the minimum weight strictly at 1.0 and boost the rest.
    valid_weights = [w for w in weights.values() if w > 0]
    if valid_weights:
        min_w = min(valid_weights)
        for c in weights:
            if weights[c] > 0:
                # Divide by min_w so majority class = 1.0 exactly
                # Cap the boosted minority classes at max_weight_cap (e.g. 2.5)
                weights[c] = min(weights[c] / min_w, max_weight_cap)

    max_observed_w = max([w for w in weights.values() if w > 0], default=1.0)

    # Fill zero-count classes with the max observed weight cap
    for c in target_classes:
        if weights[int(c)] < 0:
            weights[int(c)] = float(min(max_observed_w, max_weight_cap))

    return weights
