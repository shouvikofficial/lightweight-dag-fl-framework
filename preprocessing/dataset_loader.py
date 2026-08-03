import os

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.model import get_preprocess_input


# ============================================
# CONFIGURATION
# ============================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "MEL",
    "NV",
    "BKL",
    "BCC",
]


# ============================================
# DULLRAZOR HAIR REMOVAL & SHADES-OF-GRAY COLOR CONSTANCY
# ============================================

def shades_of_gray_cv(image_np: np.ndarray, power: int = 6) -> np.ndarray:
    """
    Applies Shades-of-Gray (Minkowski p-norm, p=6) color constancy to normalize
    camera lighting and white-balance skin lesion images.
    """
    try:
        img_float = image_np.astype(np.float32)
        illum_e = np.power(np.mean(np.power(np.abs(img_float), power), axis=(0, 1)), 1.0 / power)
        illum_e = np.where(illum_e == 0, 1e-6, illum_e)
        scale = np.mean(illum_e)
        normalized = img_float / illum_e * scale
        return np.clip(normalized, 0, 255).astype(np.float32)
    except Exception:
        return image_np.astype(np.float32)


def remove_hair_cv(image_np: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        return image_np.astype(np.float32)
    try:
        image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(image_uint8, hair_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        return inpainted.astype(np.float32)
    except Exception:
        return image_np.astype(np.float32)



# ============================================
# METADATA PROCESSING & ALIGNMENT
# ============================================

_CACHED_META_LOOKUP = None

def load_and_preprocess_metadata(meta_csv_path="dataset/raw/ISIC_2019_Training_Metadata.csv") -> dict:
    """
    Loads and preprocesses ISIC 2019 metadata (age, sex, anatom_site).
    Normalizes age, maps sex and anatomical site to categorical indices.
    Returns dict mapping image_id -> np.ndarray([age_norm, sex_enc, site_enc]).
    """
    global _CACHED_META_LOOKUP
    if _CACHED_META_LOOKUP is not None:
        return _CACHED_META_LOOKUP

    search_paths = [
        meta_csv_path,
        "dataset/raw/ISIC_2019_Training_Metadata.csv",
        "dataset/ISIC_2019_Training_Metadata.csv",
    ]
    csv_file = None
    for p in search_paths:
        if os.path.exists(p):
            csv_file = p
            break

    if not csv_file:
        print("[METADATA] ⚠️ Metadata CSV not found — using zero fallback vectors.")
        _CACHED_META_LOOKUP = {}
        return _CACHED_META_LOOKUP

    df = pd.read_csv(csv_file)
    df["image"] = df["image"].astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)

    # Impute missing values
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median() if len(df["age_approx"].dropna()) > 0 else 50.0)
    df["sex"] = df["sex"].fillna("unknown")
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")

    # Vocab encodings
    sex_map = {"female": 0.0, "male": 1.0, "unknown": 2.0}
    site_vocab = sorted(df["anatom_site_general"].unique())
    site_map = {v: float(i) for i, v in enumerate(site_vocab)}

    age_mean = df["age_approx"].mean()
    age_std = df["age_approx"].std() if df["age_approx"].std() > 0 else 1.0

    df["age_norm"] = (df["age_approx"] - age_mean) / age_std
    df["sex_enc"] = df["sex"].map(lambda x: sex_map.get(str(x).lower(), 2.0))
    df["site_enc"] = df["anatom_site_general"].map(lambda x: site_map.get(x, 0.0))

    meta_dict = {}
    for _, row in df.iterrows():
        meta_dict[row["image"]] = np.array([row["age_norm"], row["sex_enc"], row["site_enc"]], dtype=np.float32)

    _CACHED_META_LOOKUP = meta_dict
    print(f"[METADATA] ✅ Loaded metadata for {len(meta_dict)} images.")
    return meta_dict


class DualInputGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence wrapper that yields Dual Inputs:
    ((image_batch, metadata_batch), label_batch)
    """
    def __init__(self, base_gen, meta_lookup):
        self.base_gen = base_gen
        self.meta_lookup = meta_lookup
        self.filenames = [os.path.splitext(os.path.basename(fn))[0] for fn in base_gen.filenames]
        self.class_indices = getattr(base_gen, "class_indices", {})
        self.n = getattr(base_gen, "n", len(self.filenames))
        self.batch_size = getattr(base_gen, "batch_size", BATCH_SIZE)

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        x_img, y = self.base_gen[idx]
        batch_size = len(x_img)
        start_i = idx * self.batch_size
        batch_fns = self.filenames[start_i : start_i + batch_size]

        meta_batch = []
        for fn in batch_fns:
            vec = self.meta_lookup.get(fn, np.array([0.0, 2.0, 0.0], dtype=np.float32))
            meta_batch.append(vec)

        meta_batch = np.array(meta_batch, dtype=np.float32)
        return (x_img, meta_batch), y

    def on_epoch_end(self):
        if hasattr(self.base_gen, "on_epoch_end"):
            self.base_gen.on_epoch_end()

    def reset(self):
        if hasattr(self.base_gen, "reset"):
            self.base_gen.reset()


# ============================================
# LOAD IMAGE
# ============================================

def load_image(image_path, image_size=IMAGE_SIZE):
    if HAS_CV2:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        image = remove_hair_cv(image)
        return image.astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        return np.array(img, dtype=np.float32)


# ============================================
# DUAL INPUT CLIENT GENERATOR PIPELINE
# ============================================

def prepare_client_generators(
    csv_path,
    image_root,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    seed=42,
    model_name="densenet121",
    enable_multimodal=True,
):
    df = pd.read_csv(csv_path)

    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'image' and 'label' columns")

    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=seed,
        stratify=df["label"],
    )

    base_prep = get_preprocess_input(model_name)
    def combined_prep(img):
        img = shades_of_gray_cv(img)
        img = remove_hair_cv(img)
        return base_prep(img)

    train_datagen = ImageDataGenerator(
        preprocessing_function=combined_prep,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=combined_prep,
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    class_names = [
        name for name, _ in sorted(
            train_gen.class_indices.items(),
            key=lambda kv: kv[1]
        )
    ]

    meta_lookup = load_and_preprocess_metadata()
    if enable_multimodal and meta_lookup:
        train_gen = DualInputGenerator(train_gen, meta_lookup)
        val_gen = DualInputGenerator(val_gen, meta_lookup)

    return train_gen, val_gen, class_names


# ============================================
# GLOBAL TEST GENERATOR
# ============================================

def prepare_global_test_generator(
    csv_path,
    image_root,
    batch_size=BATCH_SIZE,
    model_name="densenet121",
    enable_multimodal=True,
):
    df = pd.read_csv(csv_path)

    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'image' and 'label' columns")

    base_prep = get_preprocess_input(model_name)
    def combined_prep(img):
        img = remove_hair_cv(img)
        return base_prep(img)

    test_datagen = ImageDataGenerator(
        preprocessing_function=combined_prep,
    )

    test_gen = test_datagen.flow_from_dataframe(
        df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    meta_lookup = load_and_preprocess_metadata()
    if enable_multimodal and meta_lookup:
        test_gen = DualInputGenerator(test_gen, meta_lookup)

    return test_gen