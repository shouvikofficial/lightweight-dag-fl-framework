import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ============================================
# BUILD MODEL
# ============================================

def build_model(
    input_shape=(224, 224, 3),
    num_classes=8,
    dropout_rate_head=0.4,
    dropout_rate_dense=0.3,
    l2_strength=1e-3,
    label_smoothing=0.05,
    learning_rate=1e-4,
    auc_name="auc",
):

    # ========================================
    # LOAD PRETRAINED EFFICIENTNETB0
    # ========================================

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        name="backbone",
    )

    # ========================================
    # FREEZE BACKBONE INITIALLY
    # ========================================

    base_model.trainable = False

    # ========================================
    # CUSTOM CLASSIFICATION HEAD
    # ========================================

    x = base_model.output

    # Hybrid pooling to capture complementary spatial statistics
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = Concatenate()([gap, gmp])

    x = BatchNormalization()(x)

    x = Dropout(dropout_rate_head)(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(l2_strength),
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(dropout_rate_dense)(x)

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(l2_strength),
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(dropout_rate_dense)(x)

    output = Dense(
        num_classes,
        activation="softmax"
    )(x)

    # ========================================
    # FINAL MODEL
    # ========================================

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    # ========================================
    # COMPILE MODEL
    # ========================================

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name=auc_name, multi_label=False),
        ],
    )

    return model


# ============================================
# UNFREEZE MODEL FOR FINE-TUNING
# ============================================

def unfreeze_model(
    model,
    fine_tune_at=100,
    learning_rate=1e-5,
    keep_batch_norm_frozen=True,
):

    # Unfreeze backbone
    backbone = model.get_layer("backbone")
    backbone.trainable = True

    # Freeze early layers
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    if keep_batch_norm_frozen:
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    # Recompile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )

    return model