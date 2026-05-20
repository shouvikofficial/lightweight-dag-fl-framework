import numpy as np


# ============================================
# MODEL QUANTIZATION
# ============================================

class Quantizer:

    def __init__(self):

        pass

    # ========================================
    # INT8 QUANTIZATION
    # ========================================

    def quantize(
        self,
        weights
    ):

        quantized_weights = []

        scales = []

        for layer in weights:

            scale = np.max(np.abs(layer)) / 127

            quantized = (
                layer / scale
            ).astype(np.int8)

            quantized_weights.append(
                quantized
            )

            scales.append(scale)

        return quantized_weights, scales

    # ========================================
    # DEQUANTIZATION
    # ========================================

    def dequantize(
        self,
        quantized_weights,
        scales
    ):

        restored = []

        for layer, scale in zip(
            quantized_weights,
            scales
        ):

            restored.append(
                layer.astype(np.float32) * scale
            )

        return restored