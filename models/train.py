from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf


# ============================================
# TRAIN FUNCTION
# ============================================

def train_one_epoch(
    model,
    x_train,
    y_train,
    batch_size=32,
    class_weight=None,
    callbacks: Optional[Iterable[tf.keras.callbacks.Callback]] = None,
    return_history: bool = False,
):
    """
    Train the Keras model for one epoch.

    Parameters
    ----------
    model        : compiled tf.keras.Model
    x_train      : numpy array of training images
    y_train      : one-hot encoded labels
    batch_size   : int
    class_weight : optional dict of class weights

    Returns
    -------
    epoch_loss     : float
    epoch_accuracy : float
    """

    history = model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weight,
        callbacks=list(callbacks) if callbacks else None,
    )

    epoch_loss = history.history["loss"][0]
    epoch_accuracy = history.history["accuracy"][0]

    if return_history:
        return epoch_loss, epoch_accuracy, history

    return epoch_loss, epoch_accuracy


# ============================================
# TRAIN WITH GENERATOR
# ============================================

def train_with_generator(
    model,
    train_generator,
    steps_per_epoch=None,
    class_weight=None,
    epochs: int = 1,
    validation_data=None,
    validation_steps=None,
    callbacks: Optional[Iterable[tf.keras.callbacks.Callback]] = None,
    return_history: bool = False,
):
    """
    Train using a Keras ImageDataGenerator.

    Parameters
    ----------
    model            : compiled tf.keras.Model
    train_generator  : Keras data generator
    steps_per_epoch  : int or None (auto-computed)
    class_weight     : optional dict of class weights

    Returns
    -------
    epoch_loss     : float
    epoch_accuracy : float
    """

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        class_weight=class_weight,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=list(callbacks) if callbacks else None,
    )

    epoch_loss = history.history["loss"][0]
    epoch_accuracy = history.history["accuracy"][0]

    if return_history:
        return epoch_loss, epoch_accuracy, history

    return epoch_loss, epoch_accuracy