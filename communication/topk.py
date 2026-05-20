import numpy as np


# ============================================
# TOP-K SPARSIFICATION
# ============================================

class TopKCompressor:

    def __init__(
        self,
        k_ratio=0.1
    ):

        self.k_ratio = k_ratio

    # ========================================
    # APPLY TOP-K
    # ========================================

    def compress(
        self,
        weights
    ):

        compressed_layers = []

        for layer in weights:

            flat = layer.flatten()

            k = max(
                1,
                int(len(flat) * self.k_ratio)
            )

            indices = np.argpartition(
                np.abs(flat),
                -k
            )[-k:]

            sparse = np.zeros_like(flat)

            sparse[indices] = flat[indices]

            compressed_layers.append(
                sparse.reshape(layer.shape)
            )

        return compressed_layers