import numpy as np


# ============================================
# MODEL WEIGHT COMPRESSION
# ============================================

class Compressor:

    def __init__(self):

        pass

    # ========================================
    # FLOAT16 COMPRESSION
    # ========================================

    def compress_weights(
        self,
        weights
    ):

        compressed = []

        for layer in weights:

            compressed.append(
                layer.astype(np.float16)
            )

        return compressed

    # ========================================
    # DECOMPRESSION
    # ========================================

    def decompress_weights(
        self,
        compressed_weights
    ):

        decompressed = []

        for layer in compressed_weights:

            decompressed.append(
                layer.astype(np.float32)
            )

        return decompressed