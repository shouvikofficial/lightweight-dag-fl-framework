from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


# ============================================
# IMAGE PREPROCESSING
# ============================================

IMAGE_SIZE = 224


def preprocess_image_for_prediction(
    image_path: str,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> np.ndarray:
    """
    Load, resize, and normalize an image for inference.

    Parameters
    ----------
    image_path : str — path to the image file

    Returns
    -------
    image : numpy array of shape (224, 224, 3), float32
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # Handle grayscale and convert BGR → RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(
    model,
    image_path: str,
    class_names: List[str],
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> Dict[str, object]:
    """
    Predict the class of a single image using a Keras model.

    Parameters
    ----------
    model       : compiled tf.keras.Model
    image_path  : str — path to image file
    class_names : list of class name strings

    Returns
    -------
    dict with 'predicted_class' and 'confidence'
    """

    image = preprocess_image_for_prediction(image_path, target_size=target_size)

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    # Run inference
    probabilities = model.predict(image, verbose=0)

    if len(class_names) != probabilities.shape[1]:
        raise ValueError(
            "class_names length does not match model output classes"
        )

    predicted_index = int(np.argmax(probabilities, axis=1)[0])

    confidence = float(probabilities[0][predicted_index])

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probabilities[0][i])
            for i in range(len(class_names))
        }
    }


# ============================================
# BATCH PREDICTION
# ============================================

def predict_batch(
    model,
    image_paths: Iterable[str],
    class_names: List[str],
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> List[Dict[str, object]]:
    """
    Predict classes for a list of images.

    Parameters
    ----------
    model       : compiled tf.keras.Model
    image_paths : list of image file paths
    class_names : list of class name strings

    Returns
    -------
    list of dicts, each with 'image', 'predicted_class', 'confidence'
    """

    results = []

    for path in image_paths:

        try:
            result = predict_image(
                model,
                path,
                class_names,
                target_size=target_size,
            )
            result["image"] = path
            results.append(result)

        except Exception as e:
            results.append({
                "image": path,
                "error": str(e)
            })

    return results