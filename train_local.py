"""
=====================================
 STANDALONE LOCAL TRAINING SCRIPT
=====================================

Use this to train the model on a single machine
WITHOUT federated learning (good for testing).

Usage:
    python train_local.py
    python train_local.py --client_id client_1 --epochs 10

This saves the trained model to:
    models/checkpoints/model.keras
"""

import os
import sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from preprocessing.dataset_loader import prepare_client_data
from preprocessing.balancing import get_class_weights
from models.model import build_model, unfreeze_model
from models.evaluate import evaluate_model


# ============================================
# CONFIG
# ============================================

DATASET_DIR   = "dataset/partitions"
IMAGE_ROOT    = "dataset/raw/ISIC_2019_Training_Input"
CHECKPOINT_DIR = "models/checkpoints"
CLASS_NAMES   = ["MEL", "NV", "BKL", "BCC", "AK", "VASC", "DF", "SCC"]


# ============================================
# ARGUMENTS
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 on ISIC 2019 locally"
    )
    parser.add_argument(
        "--client_id",
        type=str,
        default="client_1",
        choices=["client_1", "client_2", "client_3", "client_4"],
        help="Which client's data to use (default: client_1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs after unfreezing backbone (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    return parser.parse_args()


# ============================================
# MAIN TRAINING LOOP
# ============================================

def train(args):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    csv_path = os.path.join(DATASET_DIR, f"{args.client_id}.csv")

    # ========================================
    # VALIDATE DATASET
    # ========================================

    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Dataset CSV not found: {csv_path}")
        print(f"Run first:  python preprocessing/partition.py")
        sys.exit(1)

    if not os.path.exists(IMAGE_ROOT):
        print(f"\n[ERROR] Image folder not found: {IMAGE_ROOT}")
        print(f"Download ISIC 2019 Training Input and place it at: {IMAGE_ROOT}")
        sys.exit(1)

    # ========================================
    # LOAD DATA
    # ========================================

    print("\n" + "=" * 50)
    print("   STANDALONE LOCAL TRAINING")
    print("=" * 50)
    print(f"  Client     : {args.client_id}")
    print(f"  CSV        : {csv_path}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Fine-tune  : {args.finetune_epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print("=" * 50 + "\n")

    print("[1/4] Loading dataset...")

    train_gen, val_gen, encoder = prepare_client_data(
        csv_path=csv_path,
        image_root=IMAGE_ROOT
    )

    # Collect all data into arrays
    x_train_list, y_train_list = [], []
    for xb, yb in train_gen:
        x_train_list.append(xb)
        y_train_list.append(yb)

    x_val_list, y_val_list = [], []
    for xb, yb in val_gen:
        x_val_list.append(xb)
        y_val_list.append(yb)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_val   = np.concatenate(x_val_list,   axis=0)
    y_val   = np.concatenate(y_val_list,   axis=0)

    print(f"    Train: {len(x_train)} samples")
    print(f"    Val  : {len(x_val)} samples")
    print(f"    Classes: {list(encoder.classes_)}\n")

    # ========================================
    # CLASS WEIGHTS
    # ========================================

    class_weights = get_class_weights(y_train)
    print(f"[2/4] Class weights computed for {len(class_weights)} classes\n")

    # ========================================
    # BUILD MODEL
    # ========================================

    print("[3/4] Building model (EfficientNetB0)...")

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=len(CLASS_NAMES)
    )

    model.summary(line_length=80, print_fn=lambda x: print(f"    {x}"))

    # ========================================
    # CALLBACKS
    # ========================================

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # ========================================
    # PHASE 1: TRAIN HEAD ONLY
    # ========================================

    print(f"\n[4/4] Phase 1: Training classification head ({args.epochs} epochs)...\n")

    history1 = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ========================================
    # PHASE 2: FINE-TUNE BACKBONE
    # ========================================

    if args.finetune_epochs > 0:

        print(f"\n[Fine-tune] Unfreezing backbone for {args.finetune_epochs} epochs...\n")

        model = unfreeze_model(model, fine_tune_at=100)

        history2 = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

    # ========================================
    # FINAL EVALUATION
    # ========================================

    print("\n[Evaluation] Running final evaluation...\n")

    results = evaluate_model(
        model=model,
        x_test=x_val,
        y_test=y_val,
        class_names=list(encoder.classes_),
        batch_size=args.batch_size
    )

    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Loss     : {results['loss']:.4f}")
    print(f"  Accuracy : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1 : {results['macro_f1']:.4f}")
    print(f"{'='*50}")
    print(f"\n{results['report']}")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.keras")
    model.save(final_path)
    print(f"\n✅ Model saved to: {final_path}")
    print(f"✅ Best model at : {CHECKPOINT_DIR}/model.keras")


if __name__ == "__main__":
    args = parse_args()
    train(args)
