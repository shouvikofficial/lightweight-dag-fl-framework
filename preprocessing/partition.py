import os
import random
import pandas as pd
from collections import defaultdict


# ============================================
# CONFIG
# ============================================

CSV_PATH = "dataset/raw/ISIC_2019_Training_GroundTruth.csv"

OUTPUT_DIR = "dataset/partitions"

CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
    "client_4"
]

DOMINANT_CLASSES = {
    "client_1": "MEL",
    "client_2": "NV",
    "client_3": "BKL",
    "client_4": None
}

DOMINANT_RATIO = 0.7

CLASS_NAMES = [
    "MEL",
    "NV",
    "BKL",
    "BCC",
    "AK",
    "VASC",
    "DF",
    "SCC"
]


# ============================================
# PARTITION FUNCTION
# ============================================

def create_partitions(
    csv_path=CSV_PATH,
    output_dir=OUTPUT_DIR,
    clients=None,
    dominant_classes=None,
    dominant_ratio=DOMINANT_RATIO,
    seed=42
):
    """
    Create Semi Non-IID federated data partitions
    from the ISIC 2019 dataset CSV.

    Parameters
    ----------
    csv_path        : path to ISIC ground-truth CSV
    output_dir      : directory to save client CSVs
    clients         : list of client names
    dominant_classes: dict mapping client → dominant class (or None)
    dominant_ratio  : fraction of dominant class per biased client
    seed            : random seed for reproducibility
    """

    if clients is None:
        clients = CLIENTS

    if dominant_classes is None:
        dominant_classes = DOMINANT_CLASSES

    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # ========================================
    # LOAD DATASET
    # ========================================

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[ERROR] CSV not found: {csv_path}\n"
            f"Please download the ISIC 2019 dataset first."
        )

    df = pd.read_csv(csv_path)

    df["image"] = df["image"] + ".jpg"

    # ========================================
    # GROUP IMAGES BY CLASS
    # ========================================

    class_images = defaultdict(list)

    for _, row in df.iterrows():

        image_name = row["image"]

        label = None

        for class_name in CLASS_NAMES:

            if row[class_name] == 1:

                label = class_name
                break

        if label:

            class_images[label].append(
                (image_name, label)
            )

    # ========================================
    # SHUFFLE CLASS IMAGES
    # ========================================

    for class_name in CLASS_NAMES:

        random.shuffle(class_images[class_name])

    # ========================================
    # CREATE CLIENT PARTITIONS
    # ========================================

    for client in clients:

        print(f"\n[INFO] Creating {client}")

        dominant_class = dominant_classes.get(client)

        client_data = []

        # ======================================
        # BALANCED CLIENT
        # ======================================

        if dominant_class is None:

            for class_name in CLASS_NAMES:

                images = class_images[class_name]

                take_count = len(images) // 10

                selected = images[:take_count]

                class_images[class_name] = images[take_count:]

                client_data.extend(selected)

        # ======================================
        # SEMI NON-IID CLIENTS
        # ======================================

        else:

            for class_name in CLASS_NAMES:

                images = class_images[class_name]

                if class_name == dominant_class:

                    take_count = int(
                        len(images) * dominant_ratio
                    )

                else:

                    take_count = int(
                        len(images)
                        * (
                            (1 - dominant_ratio)
                            / (len(CLASS_NAMES) - 1)
                        )
                    )

                selected = images[:take_count]

                class_images[class_name] = images[take_count:]

                client_data.extend(selected)

        # ======================================
        # SAVE CLIENT CSV
        # ======================================

        client_df = pd.DataFrame(
            client_data,
            columns=["image", "label"]
        )

        save_path = os.path.join(
            output_dir,
            f"{client}.csv"
        )

        client_df.to_csv(
            save_path,
            index=False
        )

        print(
            f"[INFO] Saved {len(client_data)} "
            f"samples to {save_path}"
        )

    print("\n====================================")
    print(" Semi Non-IID Partition Completed ")
    print("====================================")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":

    create_partitions()