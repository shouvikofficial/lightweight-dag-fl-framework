import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)


# ============================================
# CONFIGURATION
# ============================================

IMAGE_SIZE = 224

BATCH_SIZE = 16


# ============================================
# LOAD IMAGE
# ============================================

def load_image(
    image_path,
    image_size=IMAGE_SIZE
):

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            f"Could not load image: "
            f"{image_path}"
        )

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ============================================
# LOAD CLIENT DATASET
# ============================================

def load_client_dataset(
    csv_path,
    image_root
):

    df = pd.read_csv(csv_path)

    images = []

    labels = []

    print(
        f"\n[INFO] Loading dataset "
        f"from {csv_path}"
    )

    for _, row in df.iterrows():

        image_name = row["image"]

        label = row["label"]

        image_path = os.path.join(
            image_root,
            image_name
        )

        try:

            image = load_image(
                image_path
            )

            images.append(image)

            labels.append(label)

        except Exception as e:

            print(
                f"[WARNING] Skipping "
                f"{image_name}: {e}"
            )

    # ========================================
    # NUMPY CONVERSION
    # ========================================

    images = np.array(images)

    labels = np.array(labels)

    # ========================================
    # LABEL ENCODING
    # ========================================

    encoder = LabelEncoder()

    encoded_labels = encoder.fit_transform(
        labels
    )

    categorical_labels = to_categorical(
        encoded_labels
    )

    print(
        f"[INFO] Loaded "
        f"{len(images)} images"
    )

    return (
        images,
        categorical_labels,
        encoder
    )


# ============================================
# TRAIN / VALIDATION SPLIT
# ============================================

def create_data_splits(
    images,
    labels,
    test_size=0.2
):

    x_train, x_val, y_train, y_val = (
        train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=np.argmax(
                labels,
                axis=1
            )
        )
    )

    return (
        x_train,
        x_val,
        y_train,
        y_val
    )


# ============================================
# IMAGE AUGMENTATION
# ============================================

def create_generators(
    x_train,
    y_train,
    x_val,
    y_val,
    batch_size=BATCH_SIZE
):

    train_datagen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.1,

        height_shift_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

        vertical_flip=True
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_generator = val_datagen.flow(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_generator,
        val_generator
    )


# ============================================
# FULL CLIENT PIPELINE
# ============================================

def prepare_client_data(
    csv_path,
    image_root
):

    images, labels, encoder = (
        load_client_dataset(
            csv_path,
            image_root
        )
    )

    (
        x_train,
        x_val,
        y_train,
        y_val
    ) = create_data_splits(
        images,
        labels
    )

    (
        train_generator,
        val_generator
    ) = create_generators(
        x_train,
        y_train,
        x_val,
        y_val
    )

    return (
        train_generator,
        val_generator,
        encoder
    )


# ============================================
# CLIENT GENERATOR PIPELINE (MEMORY-SAFE)
# ============================================

def prepare_client_generators(
    csv_path,
    image_root,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    seed=42,
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

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    class_names = [
        name for name, _ in sorted(
            train_generator.class_indices.items(),
            key=lambda kv: kv[1]
        )
    ]

    return train_generator, val_generator, class_names


# ============================================
# GLOBAL TEST GENERATOR
# ============================================

def prepare_global_test_generator(
    csv_path,
    image_root,
    batch_size=BATCH_SIZE,
):
    df = pd.read_csv(csv_path)

    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'image' and 'label' columns")

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_dataframe(
        df,
        directory=image_root,
        x_col="image",
        y_col="label",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )

    return test_generator