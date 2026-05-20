import numpy as np


# ============================================
# GRADIENT CLIPPING
# ============================================

class GradientClipper:

    def __init__(
        self,
        clip_value=1.0
    ):

        self.clip_value = clip_value

    # ========================================
    # CLIP GRADIENTS
    # ========================================

    def clip(
        self,
        gradients
    ):

        clipped_gradients = []

        for grad in gradients:

            clipped = np.clip(
                grad,
                -self.clip_value,
                self.clip_value
            )

            clipped_gradients.append(
                clipped
            )

        return clipped_gradients