import os
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================
# CONFIG
# ============================================

IMAGE_SIZE = 224


# ============================================
# LOAD CLIENT CSV
# ============================================

def load_client_dataframe(csv_path):

    df = pd.read_csv(csv_path)

    return df


# ============================================
# LOAD AND PREPROCESS IMAGE
# ============================================

def preprocess_image(
    image_path,
    image_size=IMAGE_SIZE
):

    # Read image
    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            f"Image not found: {image_path}"
        )

    # Convert BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    # EfficientNetB0 has an internal Rescaling layer trained on [0, 255].
    # preprocess_input converts uint8 [0,255] -> float32 [-1, 1] as expected.
    image = image.astype("float32")
    image = preprocess_input(image)

    return image


# ============================================
# LOAD CLIENT DATASET
# ============================================

def load_client_dataset(
    csv_path,
    image_root
):

    df = load_client_dataframe(csv_path)

    images = []
    labels = []

    for _, row in df.iterrows():

        image_name = row["image"]

        label = row["label"]

        image_path = os.path.join(
            image_root,
            image_name
        )

        try:

            image = preprocess_image(image_path)

            images.append(image)

            labels.append(label)

        except Exception as e:

            print(
                f"[WARNING] Skipping "
                f"{image_name}: {e}"
            )

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

    # ========================================
    # CONVERT TO NUMPY
    # ========================================

    images = np.array(images)

    labels = np.array(categorical_labels)

    return (
        images,
        labels,
        encoder
    )