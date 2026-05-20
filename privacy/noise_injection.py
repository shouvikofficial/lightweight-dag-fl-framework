import numpy as np


# ============================================
# NOISE INJECTION
# ============================================

class NoiseInjector:

    def __init__(
        self,
        noise_std=0.005
    ):

        self.noise_std = noise_std

    # ========================================
    # INJECT NOISE
    # ========================================

    def inject(
        self,
        weights
    ):

        noisy_weights = []

        for layer in weights:

            noise = np.random.normal(
                0,
                self.noise_std,
                layer.shape
            )

            noisy_layer = layer + noise

            noisy_weights.append(
                noisy_layer
            )

        return noisy_weights