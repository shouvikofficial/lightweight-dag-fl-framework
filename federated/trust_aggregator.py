"""
===========================================================
 MULTI-FACTOR ADAPTIVE TRUST-AWARE AGGREGATOR
===========================================================

Computes dynamic Trust Scores using a 4-Factor Adaptive Formula:
  Trust = 0.35 * Historical_Trust 
        + 0.25 * Model_Similarity 
        + 0.20 * Validation_Accuracy 
        + 0.20 * Blockchain_Verification

Implements a 3-Tier Security Action System:
  - Trust >= 0.8  : ACCEPT  (Full aggregation weight)
  - 0.5 <= T < 0.8: PENALIZE (Reduced aggregation weight)
  - Trust < 0.5   : REJECT  (Excluded completely from global update)

Supports Adaptive Trust Decay & Gradual Recovery across FL rounds.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """Flattens a list of weight matrices into a single 1D vector."""
    return np.concatenate([w.flatten() for w in weights])


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class TrustAwareAggregator:
    """
    4-Factor Adaptive Trust-Aware Aggregator with 3-Tier Security Decision System.

    Weights:
        w_hist = 0.35  (Historical Trust)
        w_sim  = 0.25  (Model Cosine Similarity)
        w_acc  = 0.20  (Validation Accuracy)
        w_bc   = 0.20  (Blockchain Verification)
    """

    def __init__(
        self,
        w_hist: float = 0.35,
        w_sim: float = 0.25,
        w_acc: float = 0.20,
        w_bc: float = 0.20,
        accept_threshold: float = 0.80,
        reject_threshold: float = 0.50,
    ):
        self.w_hist = w_hist
        self.w_sim = w_sim
        self.w_acc = w_acc
        self.w_bc = w_bc
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

        self.client_trust_scores: Dict[str, float] = {}

    def compute_trust_scores(
        self,
        client_ids: List[str],
        client_weights: List[List[np.ndarray]],
        client_accuracies: Optional[Dict[str, float]] = None,
        blockchain_validations: Optional[Dict[str, bool]] = None,
        prev_global_weights: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes 4-Factor Dynamic Trust Score for each client.

        Returns:
            Dict mapping client_id -> {
                "similarity": float,
                "val_accuracy": float,
                "blockchain_valid": bool,
                "prev_trust": float,
                "trust_score": float,
                "action": str ("ACCEPT", "PENALIZE", "REJECT")
            }
        """
        if not client_weights or not client_ids:
            return {}

        # Default maps if optional params not provided
        client_accuracies = client_accuracies or {cid: 0.80 for cid in client_ids}
        blockchain_validations = blockchain_validations or {cid: True for cid in client_ids}

        # Compute weight update delta ΔW = W_client - W_global
        deltas = []
        for weights in client_weights:
            if prev_global_weights is not None:
                delta = [w - g for w, g in zip(weights, prev_global_weights)]
            else:
                delta = weights
            deltas.append(_flatten_weights(delta))

        # Reference vector: elementwise median across all client deltas
        deltas_matrix = np.vstack(deltas)
        median_delta = np.median(deltas_matrix, axis=0)

        results = {}
        for cid, delta_vec in zip(client_ids, deltas):
            # Factor 1: Model Similarity (Cosine mapped to [0, 1])
            sim = _cosine_similarity(delta_vec, median_delta)
            scaled_sim = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

            # Factor 2: Validation Accuracy (normalized [0, 1])
            acc = float(np.clip(client_accuracies.get(cid, 0.5), 0.0, 1.0))

            # Factor 3: Blockchain Verification Score (1.0 if valid, 0.0 if invalid)
            bc_valid = bool(blockchain_validations.get(cid, True))
            bc_score = 1.0 if bc_valid else 0.0

            # Factor 4: Historical Trust (defaults to 1.0 for new clients)
            prev_trust = self.client_trust_scores.get(cid, 1.0)

            # 4-Factor Dynamic Trust Formula
            new_trust = (
                self.w_hist * prev_trust
                + self.w_sim * scaled_sim
                + self.w_acc * acc
                + self.w_bc * bc_score
            )
            new_trust = float(np.clip(new_trust, 0.0, 1.0))

            # Update adaptive historical trust state
            self.client_trust_scores[cid] = new_trust

            # 3-Tier Security Decision
            if new_trust >= self.accept_threshold:
                action = "ACCEPT"
            elif new_trust >= self.reject_threshold:
                action = "PENALIZE"
            else:
                action = "REJECT"

            results[cid] = {
                "similarity": float(sim),
                "scaled_similarity": scaled_sim,
                "val_accuracy": acc,
                "blockchain_valid": bc_valid,
                "prev_trust": prev_trust,
                "trust_score": new_trust,
                "action": action,
            }

        return results

    def aggregate(
        self,
        client_ids: List[str],
        client_weights: List[List[np.ndarray]],
        client_sizes: List[int],
        client_accuracies: Optional[Dict[str, float]] = None,
        blockchain_validations: Optional[Dict[str, bool]] = None,
        prev_global_weights: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Dict[str, float]]]:
        """
        Aggregates parameters using dynamic trust weighting.
        3-Tier Weight Adjustment:
          - ACCEPT   : full weight (Trust * Size)
          - PENALIZE : 50% weight (Trust * 0.5 * Size)
          - REJECT   : 0% weight (excluded)
        """
        trust_info = self.compute_trust_scores(
            client_ids=client_ids,
            client_weights=client_weights,
            client_accuracies=client_accuracies,
            blockchain_validations=blockchain_validations,
            prev_global_weights=prev_global_weights,
        )

        effective_weights = []
        valid_indices = []

        for i, cid in enumerate(client_ids):
            info = trust_info[cid]
            action = info["action"]
            t_score = info["trust_score"]
            size = client_sizes[i]

            if action == "ACCEPT":
                w_eff = t_score * size
                effective_weights.append(w_eff)
                valid_indices.append(i)
            elif action == "PENALIZE":
                w_eff = t_score * 0.5 * size  # 50% penalty on weight
                effective_weights.append(w_eff)
                valid_indices.append(i)
            else:  # REJECT
                pass

        # Fallback if all clients rejected: use highest trust client
        if not valid_indices:
            best_cid = max(trust_info.keys(), key=lambda k: trust_info[k]["trust_score"])
            best_idx = client_ids.index(best_cid)
            valid_indices = [best_idx]
            effective_weights = [trust_info[best_cid]["trust_score"] * client_sizes[best_idx]]
            trust_info[best_cid]["action"] = "PENALIZE (FALLBACK)"

        total_weight = sum(effective_weights)
        norm_weights = [w / total_weight for w in effective_weights]

        # Perform weighted layer-wise aggregation
        first_layers = client_weights[valid_indices[0]]
        aggregated = []
        for layer_idx in range(len(first_layers)):
            l_shape = first_layers[layer_idx].shape
            l_dtype = first_layers[layer_idx].dtype
            weighted_layer = np.zeros(l_shape, dtype=np.float64)

            for idx, i in enumerate(valid_indices):
                w_layer = client_weights[i][layer_idx].astype(np.float64)
                weighted_layer += w_layer * norm_weights[idx]

            aggregated.append(weighted_layer.astype(l_dtype))

        return aggregated, trust_info
