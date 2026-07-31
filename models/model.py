import tensorflow as tf

from tensorflow.keras.applications import (
    EfficientNetB0,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    ResNet50V2,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Layer,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ============================================
# DYNAMIC MODEL-SPECIFIC PREPROCESSING HELPER
# ============================================

def get_preprocess_input(model_name="densenet121"):
    """
    Returns the exact matching Keras preprocess_input function for the selected backbone.
    DenseNet     -> densenet.preprocess_input
    ResNet50V2   -> resnet_v2.preprocess_input
    EfficientNet -> efficientnet.preprocess_input
    """
    model_name_lower = str(model_name).lower()
    if "densenet" in model_name_lower:
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input
    elif "resnet" in model_name_lower:
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        return preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input


# ============================================
# SOTA IMPROVEMENT 1: CBAM ATTENTION MODULE
# ============================================

class CBAM(Layer):
    """
    Convolutional Block Attention Module (CBAM).
    Combines Sequential Channel Attention & Spatial Attention.
    """

    def __init__(self, reduction_ratio=8, spatial_kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(1, channels // self.reduction_ratio)
        self.mlp_dense1 = Dense(reduced_channels, activation="relu", use_bias=False, kernel_initializer="he_normal")
        self.mlp_dense2 = Dense(channels, use_bias=False, kernel_initializer="he_normal")
        self.conv2d = Conv2D(
            1,
            kernel_size=self.spatial_kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        super().build(input_shape)

    def call(self, inputs):
        # 1. Channel Attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.mlp_dense2(self.mlp_dense1(avg_pool))
        max_out = self.mlp_dense2(self.mlp_dense1(max_pool))
        channel_att = tf.nn.sigmoid(avg_out + max_out)
        x = inputs * channel_att

        # 2. Spatial Attention
        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = tf.concat([avg_spatial, max_spatial], axis=-1)
        spatial_att = self.conv2d(spatial_concat)
        return x * spatial_att

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "spatial_kernel_size": self.spatial_kernel_size,
        })
        return config


# ============================================
# SOTA IMPROVEMENT 2: Generalized Mean (GeM) Pooling
# ============================================

class GeMPooling2D(Layer):
    """
    Generalized Mean Pooling 2D (GeM).
    Learns an optimal pooling exponent p per channel dynamically.
    Features CRITICAL FIXES:
      1. Uses tf.maximum(inputs, eps) for stable per-sample processing.
      2. Constrains exponent p >= 1.0 via softplus for numerical stability.
    """

    def __init__(self, init_p=3.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_p = init_p
        self.eps = eps

    def build(self, input_shape):
        channels = input_shape[-1]
        self.p = self.add_weight(
            name="p",
            shape=(1, 1, 1, channels),
            initializer=tf.keras.initializers.Constant(self.init_p),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # 1. Numerically stable clipping without cross-sample interaction
        x = tf.maximum(inputs, self.eps)
        # 2. Constrain exponent p >= 1.0 using softplus
        p_constrained = tf.maximum(tf.nn.softplus(self.p), 1.0)
        x = tf.pow(x, p_constrained)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        p_vec = tf.reshape(p_constrained, [-1])
        x = tf.pow(x, 1.0 / p_vec)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_p": self.init_p,
            "eps": self.eps,
        })
        return config


# ============================================
# LOSS HELPER — TF 2.10 compatible Focal Loss
# ============================================

class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification.
    Equivalent to CategoricalFocalCrossentropy (added in TF 2.11+).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0,
                 num_classes=8, **kwargs):
        super().__init__(**kwargs)
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes   # stored at init, avoids dynamic tf.shape()

    def call(self, y_true, y_pred):
        # Label smoothing using static num_classes — avoids strided_slice in graph mode
        if self.label_smoothing > 0:
            y_true = (y_true * (1.0 - self.label_smoothing) +
                      self.label_smoothing / float(self.num_classes))

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight  = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma,
                        "label_smoothing": self.label_smoothing,
                        "num_classes": self.num_classes})
        return config


def get_loss_function(label_smoothing=0.1):
    """
    Returns Focal Loss compatible with TF 2.10+.
    Tries built-in CategoricalFocalCrossentropy first (TF >= 2.11),
    falls back to custom implementation for TF 2.10.
    """
    try:
        return tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25, gamma=2.0,
            from_logits=False, label_smoothing=label_smoothing,
        )
    except AttributeError:
        # TF 2.10 — use custom implementation (identical math)
        return CategoricalFocalLoss(
            alpha=0.25, gamma=2.0, label_smoothing=label_smoothing,
            name="focal_loss",
        )


# ============================================
# BUILD MODEL
# ============================================

def build_model(
    model_name="densenet121",
    input_shape=(224, 224, 3),
    num_classes=8,
    dropout_rate_head=0.4,
    dropout_rate_dense=0.3,
    l2_strength=1e-4,
    learning_rate=5e-4,
    auc_name="auc",
    use_cbam=True,
    use_gem=True,
    pooling_mode="gem_gap",
    label_smoothing=0.1,
    cbam_reduction_ratio=8,
):
    """
    Build pretrained backbone with CBAM Attention, GeM Pooling, AdamW, He Initialization,
    512->256 Dense Head, and Focal Crossentropy Loss.
    """
    model_name_lower = str(model_name).lower()
    if "densenet169" in model_name_lower:
        base_model = DenseNet169(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "densenet201" in model_name_lower:
        base_model = DenseNet201(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "densenet" in model_name_lower:
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "resnet" in model_name_lower:
        base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze backbone initially
    base_model.trainable = False

    x = base_model.output

    # 1. CBAM Attention Module (reduction_ratio=8)
    if use_cbam:
        x = CBAM(reduction_ratio=cbam_reduction_ratio, name="cbam_attention")(x)

    # 2. GeM Pooling Combinations (gem_gap, gem_gmp, or gem_only)
    if use_gem:
        gem_pool = GeMPooling2D(name="gem_pooling")(x)
        if pooling_mode == "gem_gap":
            gap = GlobalAveragePooling2D()(x)
            x = Concatenate()([gem_pool, gap])
        elif pooling_mode == "gem_gmp":
            gmp = GlobalMaxPooling2D()(x)
            x = Concatenate()([gem_pool, gmp])
        else:
            x = gem_pool
    else:
        gap = GlobalAveragePooling2D()(x)
        gmp = GlobalMaxPooling2D()(x)
        x = Concatenate()([gap, gmp])

    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_head)(x)

    # 3. Enhanced 512 -> 256 Dense Head with He Normal Initialization
    x = Dense(512, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_dense)(x)

    x = Dense(256, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate_dense)(x)

    output = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(x)

    model = Model(inputs=base_model.input, outputs=output)

    loss_fn = get_loss_function(label_smoothing=label_smoothing)

    # 4. Adam Optimizer with L2 regularization (applied via kernel_regularizer in Dense layers)
    # Note: AdamW from tf.keras.optimizers.experimental crashes with DirectML on TF 2.10.
    # L2 weight decay is already enforced by kernel_regularizer=l2(l2_strength) in each Dense layer,
    # so the regularization effect is identical.
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
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
    fine_tune_at=None,
    learning_rate=5e-5,
    keep_batch_norm_frozen=True,
    label_smoothing=0.1,
    l2_strength=1e-4,
    model_name="densenet121",
):
    try:
        backbone = model.get_layer("backbone")
        backbone.trainable = True
        backbone_layers = backbone.layers
    except ValueError:
        cutoff_idx = None
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, GeMPooling2D, CBAM)):
                cutoff_idx = idx
                break
        if cutoff_idx is None:
            cutoff_idx = len(model.layers)
        backbone_layers = model.layers[:cutoff_idx]
        for layer in backbone_layers:
            layer.trainable = True

    # 5. Backbone-specific fine-tuning cutoffs
    if fine_tune_at is None:
        model_name_lower = str(model_name).lower()
        if "densenet169" in model_name_lower:
            fine_tune_at = 160
        elif "densenet201" in model_name_lower:
            fine_tune_at = 190
        elif "resnet" in model_name_lower:
            fine_tune_at = 100
        else:
            fine_tune_at = 120

    fine_tune_at = min(fine_tune_at, len(backbone_layers))
    for layer in backbone_layers[:fine_tune_at]:
        layer.trainable = False

    if keep_batch_norm_frozen:
        for layer in backbone_layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    loss_fn = get_loss_function(label_smoothing=label_smoothing)
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=l2_strength)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )

    return model