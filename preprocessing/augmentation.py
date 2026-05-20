from typing import Optional

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def _random_erasing(
    img: np.ndarray,
    p: float = 0.5,
    sl: float = 0.02,
    sh: float = 0.2,
    r1: float = 0.3,
) -> np.ndarray:
    """Apply random erasing to a single image (H, W, C)."""

    if np.random.rand() > p:
        return img

    h, w, c = img.shape
    area = h * w

    for _ in range(10):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1 / r1)

        erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

        if erase_h < h and erase_w < w:
            top = np.random.randint(0, h - erase_h)
            left = np.random.randint(0, w - erase_w)
            img = img.copy()
            img[top : top + erase_h, left : left + erase_w, :] = np.random.uniform(
                0.0, 1.0, (erase_h, erase_w, c)
            )
            break

    return img


def _sample_beta(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return np.random.beta(alpha, alpha)


def _mixup_batch(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lam = _sample_beta(alpha)
    indices = np.random.permutation(x.shape[0])
    x_mix = lam * x + (1 - lam) * x[indices]
    y_mix = lam * y + (1 - lam) * y[indices]
    return x_mix, y_mix


def _cutmix_batch(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lam = _sample_beta(alpha)
    indices = np.random.permutation(x.shape[0])
    x2 = x[indices]
    y2 = y[indices]

    h, w = x.shape[1], x.shape[2]
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2b = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2b = np.clip(cy + cut_h // 2, 0, h)

    x_cut = x.copy()
    x_cut[:, y1:y2b, x1:x2b, :] = x2[:, y1:y2b, x1:x2b, :]

    box_area = (x2b - x1) * (y2b - y1)
    lam_adj = 1 - box_area / (h * w)
    y_cut = lam_adj * y + (1 - lam_adj) * y2

    return x_cut, y_cut


def _mixup_cutmix_generator(
    base_gen,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
):
    while True:
        x_batch, y_batch = next(base_gen)
        r = np.random.rand()

        if r < mixup_prob and mixup_alpha > 0:
            x_batch, y_batch = _mixup_batch(x_batch, y_batch, mixup_alpha)
        elif r < mixup_prob + cutmix_prob and cutmix_alpha > 0:
            x_batch, y_batch = _cutmix_batch(x_batch, y_batch, cutmix_alpha)

        yield x_batch, y_batch


# ============================================
# TRAIN AUGMENTATION
# ============================================

train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=10.0,
    fill_mode="nearest",
    preprocessing_function=_random_erasing,
)


# ============================================
# VALIDATION AUGMENTATION
# ============================================

val_datagen = ImageDataGenerator()


# ============================================
# GENERATOR FUNCTIONS
# ============================================

def create_train_generator(
    x_train,
    y_train,
    batch_size=16,
    seed: Optional[int] = None,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 0.2,
    mixup_prob: float = 0.5,
    cutmix_prob: float = 0.5,
):

    base_gen = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return base_gen

    return _mixup_cutmix_generator(
        base_gen,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mixup_prob=mixup_prob,
        cutmix_prob=cutmix_prob,
    )


def create_validation_generator(
    x_val,
    y_val,
    batch_size=16,
    seed: Optional[int] = None,
):

    return val_datagen.flow(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )