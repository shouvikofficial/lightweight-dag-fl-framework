import numpy as np


# ============================================
# DIFFERENTIAL PRIVACY
# ============================================

class DifferentialPrivacy:

    def __init__(
        self,
        noise_multiplier=0.01
    ):

        self.noise_multiplier = noise_multiplier

    # ========================================
    # ADD GAUSSIAN NOISE
    # ========================================

    def add_noise(
        self,
        weights
    ):

        noisy_weights = []

        for layer in weights:

            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_multiplier,
                size=layer.shape
            )

            noisy_layer = layer + noise

            noisy_weights.append(
                noisy_layer
            )

        return noisy_weights