"""
===================================================
 UMAP CLIENT FEATURE DISTRIBUTION GENERATOR
===================================================

Extracts feature embeddings from client dataset partitions (Client 1, Client 2, Client 3, Client 4)
using a trained backbone, projects them into 2D using UMAP (or t-SNE fallback),
and generates a publication-quality Client Feature Distribution plot matching journal standards.

Usage:
    python plot_umap_distribution.py
    python plot_umap_distribution.py --model_name efficientnetb0
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import zipfile

# Check for UMAP vs t-SNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

from models.model import build_model, GeMPooling2D, CBAM, CategoricalFocalLoss
from preprocessing.dataset_loader import load_image, load_and_preprocess_metadata, IMAGE_SIZE, CLASS_NAMES

DATASET_DIR = "dataset/partitions"
IMAGE_ROOT = "dataset/raw/ISIC_2019_Training_Input"
CHECKPOINT_DIR = "models/checkpoints"
PLOTS_DIR = "models/plots"

CLIENTS = ["client_1", "client_2", "client_3", "client_4"]
CLIENT_LABELS = {"client_1": "Client 1", "client_2": "Client 2", "client_3": "Client 3", "client_4": "Client 4"}
COLORS = ["#e74c3c", "#d4ac0d", "#2ecc71", "#3498db"]  # Red, Yellow, Green, Blue matching paper styling


def load_feature_extractor(model_name="efficientnetb0"):
    """Loads model and constructs a feature extractor model outputting dense feature embeddings."""
    model_tag = str(model_name).lower()
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.keras")

    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"centralized_best_{model_tag}.h5")

    custom_objs = {
        "GeMPooling2D": GeMPooling2D,
        "CBAM": CBAM,
        "CategoricalFocalLoss": CategoricalFocalLoss,
    }

    full_model = build_model(
        model_name=model_name,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=len(CLASS_NAMES),
        pooling_mode="gem_gap",
    )

    if os.path.exists(ckpt_path):
        try:
            full_model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
        except Exception:
            try:
                full_model.get_layer("backbone").load_weights(ckpt_path, by_name=True, skip_mismatch=True)
            except Exception:
                print(f"ℹ️ Loaded backbone architecture for {model_name}...")

    # Extract feature layer (visual_dense or fusion_dense before final classification output)
    feature_layer_name = None
    for target in ["fusion_dense", "visual_dense", "fusion_bn", "visual_bn"]:
        for layer in full_model.layers:
            if layer.name == target:
                feature_layer_name = target
                break
        if feature_layer_name:
            break

    if feature_layer_name:
        feature_extractor = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=full_model.get_layer(feature_layer_name).output
        )
        print(f"✅ Extracted features from layer: {feature_layer_name}")
    else:
        # Fallback to second-to-last layer
        feature_extractor = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=full_model.layers[-2].output
        )
        print(f"✅ Extracted features from layer: {full_model.layers[-2].name}")

    return feature_extractor


def extract_client_embeddings(feature_extractor, max_samples_per_client=400):
    """Extracts feature vectors for each client node's dataset partition."""
    all_embeddings = []
    all_client_ids = []

    # Use robust dataset_loader metadata lookup
    meta_lookup = load_and_preprocess_metadata()

    for client_name in CLIENTS:
        csv_path = os.path.join(DATASET_DIR, f"{client_name}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Partition not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if len(df) > max_samples_per_client:
            df = df.sample(n=max_samples_per_client, random_state=42)

        print(f"Processing {client_name} ({len(df)} samples)...")
        img_batch, meta_batch = [], []
        
        for idx, row in df.iterrows():
            img_filename = row["image"]
            img_path = os.path.join(IMAGE_ROOT, img_filename)
            
            try:
                img = load_image(img_path, image_size=IMAGE_SIZE)
                # Normalize image pixel values
                img_prep = img / 255.0
                img_batch.append(img_prep)

                clean_name = os.path.splitext(os.path.basename(img_filename))[0]
                vec = meta_lookup.get(clean_name, np.array([0.0, 0.5, 0.0], dtype=np.float32))
                meta_batch.append(vec)
            except Exception:
                continue

            if len(img_batch) >= 32 or idx == df.index[-1]:
                if len(img_batch) == 0:
                    continue
                imgs_np = np.array(img_batch, dtype=np.float32)
                metas_np = np.array(meta_batch, dtype=np.float32)
                
                features = feature_extractor([imgs_np, metas_np], training=False)
                all_embeddings.append(features.numpy())
                all_client_ids.extend([CLIENT_LABELS[client_name]] * len(features))
                
                img_batch, meta_batch = [], []

    X = np.concatenate(all_embeddings, axis=0)
    y = np.array(all_client_ids)
    return X, y


def generate_umap_plot(X, y, model_name="efficientnetb0"):
    """Applies UMAP / t-SNE and plots publication-quality Client Distribution figure."""
    print(f"\n[UMAP] Reducing {X.shape[0]} samples from {X.shape[1]}-dim space to 2D...")
    
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        X_2d = reducer.fit_transform(X)
        algo_name = "Umap"
    else:
        print("ℹ️ umap-learn not installed — falling back to t-SNE projection...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X)
        algo_name = "t-SNE"

    # Plot styling matching paper screenshot
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Plot each client scatter
    unique_clients = list(CLIENT_LABELS.values())
    palette = dict(zip(unique_clients, COLORS))

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y,
        palette=palette,
        style=y,
        s=25,
        alpha=0.85,
        edgecolor="none",
        ax=ax,
    )

    ax.set_xlabel(f"{algo_name} dimension 1", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel(f"{algo_name} dimension 2", fontsize=14, fontweight="bold", labelpad=10)
    
    # Format Legend in top-left
    leg = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        fontsize=12,
        markerscale=1.5,
    )
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_edgecolor("#cccccc")

    # Title / Caption at bottom
    plt.title(f"(f) Fed-ISIC2019 ({model_name.upper()})", fontsize=16, fontweight="bold", y=-0.18)
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"umap_client_distribution_{model_name.lower()}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\n✅ Publication figure saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP Client Feature Distribution Plot")
    parser.add_argument("--model_name", type=str, default="efficientnetb0", help="Backbone model name")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples per client node")
    args = parser.parse_args()

    print("=" * 60)
    print(" 🎨 GENERATING UMAP CLIENT FEATURE DISTRIBUTION FIGURE")
    print(f"    Model       : {args.model_name.upper()}")
    print(f"    Max Samples : {args.max_samples} per client node")
    print("=" * 60)

    extractor = load_feature_extractor(args.model_name)
    X, y = extract_client_embeddings(extractor, max_samples_per_client=args.max_samples)
    generate_umap_plot(X, y, model_name=args.model_name)


if __name__ == "__main__":
    main()
