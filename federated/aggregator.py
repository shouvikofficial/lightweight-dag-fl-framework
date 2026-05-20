import numpy as np


# ============================================
# FEDERATED AGGREGATOR
# ============================================

class Aggregator:

    def __init__(self):

        pass

    # ========================================
    # FEDAVG AGGREGATION
    # ========================================

    def fedavg(self, client_weights):

        aggregated_weights = []

        for layer_weights in zip(*client_weights):

            layer_average = np.mean(
                np.array(layer_weights),
                axis=0
            )

            aggregated_weights.append(
                layer_average
            )

        return aggregated_weights

    # ========================================
    # WEIGHTED AGGREGATION
    # ========================================

    def weighted_average(
        self,
        client_weights,
        client_sizes
    ):

        total_samples = sum(client_sizes)

        aggregated_weights = []

        for layer_weights in zip(*client_weights):

            weighted_layer = np.zeros_like(
                layer_weights[0]
            )

            for weights, size in zip(
                layer_weights,
                client_sizes
            ):

                weighted_layer += (
                    weights * (size / total_samples)
                )

            aggregated_weights.append(
                weighted_layer
            )

        return aggregated_weights