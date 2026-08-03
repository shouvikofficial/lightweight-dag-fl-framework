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
GLOBAL_TEST_RATIO = 0.1

CLASS_NAMES = [
    "MEL",
    "NV",
    "BKL",
    "BCC",
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
    seed=42,
    global_test_ratio=GLOBAL_TEST_RATIO,
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
    # RESERVE GLOBAL TEST SET
    # ========================================

    global_test = []
    for class_name in CLASS_NAMES:
        images = class_images[class_name]
        take_count = int(len(images) * global_test_ratio)
        global_test.extend(images[:take_count])
        class_images[class_name] = images[take_count:]

    global_test_df = pd.DataFrame(
        global_test,
        columns=["image", "label"],
    )
    global_test_path = os.path.join(output_dir, "global_test.csv")
    global_test_df.to_csv(global_test_path, index=False)
    print(
        f"[INFO] Saved {len(global_test)} global test samples to {global_test_path}"
    )

    # ========================================
    # CREATE CLIENT PARTITIONS
    # ========================================

    client_buffers = {client: [] for client in clients}

    for class_name in CLASS_NAMES:
        images = class_images[class_name]
        total = len(images)

        dominant_client = None
        for client, dom in dominant_classes.items():
            if dom == class_name:
                dominant_client = client
                break

        if dominant_client:
            weights = {}
            for client in clients:
                if client == dominant_client:
                    weights[client] = dominant_ratio
                else:
                    weights[client] = (1 - dominant_ratio) / (len(clients) - 1)
        else:
            weights = {client: 1 / len(clients) for client in clients}

        raw_counts = {client: weights[client] * total for client in clients}
        counts = {client: int(raw_counts[client]) for client in clients}

        remaining = total - sum(counts.values())
        for client in sorted(
            clients,
            key=lambda c: raw_counts[c] - counts[c],
            reverse=True,
        ):
            if remaining <= 0:
                break
            counts[client] += 1
            remaining -= 1

        start = 0
        for client in clients:
            take_count = counts[client]
            selected = images[start:start + take_count]
            client_buffers[client].extend(selected)
            start += take_count

    for client in clients:

        print(f"\n[INFO] Creating {client}")

        client_df = pd.DataFrame(
            client_buffers[client],
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
            f"[INFO] Saved {len(client_buffers[client])} "
            f"samples to {save_path}"
        )

    print("\n====================================")
    print(" Semi Non-IID Partition Completed ")
    print("====================================")


# ============================================
# ENTRY POINT
# ============================================

def create_demographic_partitions(
    csv_path=CSV_PATH,
    meta_path="dataset/raw/ISIC_2019_Training_Metadata.csv",
    output_dir=OUTPUT_DIR,
    global_test_ratio=GLOBAL_TEST_RATIO,
    seed=42,
):
    """
    Partitions ISIC 2019 dataset into 4 demographic hospital clients (Option 3):
      client_1: Elderly Patients (Age > 60)
      client_2: Young/Adult Patients (Age <= 40)
      client_3: Head/Neck & Upper Extremities Clinic
      client_4: Torso & Lower Extremities Clinic
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df_gt = pd.read_csv(csv_path)
    df_gt["image_clean"] = df_gt["image"].astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)

    # Extract single label
    labels = []
    for _, row in df_gt.iterrows():
        lbl = "NV"
        for c in CLASS_NAMES:
            if c in row and row[c] == 1:
                lbl = c
                break
        labels.append(lbl)
    df_gt["label"] = labels

    if not os.path.exists(meta_path):
        meta_path = "dataset/ISIC_2019_Training_Metadata.csv"

    if os.path.exists(meta_path):
        df_meta = pd.read_csv(meta_path)
        df_meta["image_clean"] = df_meta["image"].astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)
        df = pd.merge(df_gt, df_meta[["image_clean", "age_approx", "sex", "anatom_site_general"]], on="image_clean", how="left")
    else:
        df = df_gt.copy()
        df["age_approx"] = 50.0
        df["sex"] = "unknown"
        df["anatom_site_general"] = "unknown"

    df["image"] = df["image_clean"] + ".jpg"

    # Reserve Global Test set (Stratified)
    test_rows, train_rows = [], []
    for cname in CLASS_NAMES:
        cdf = df[df["label"] == cname].sample(frac=1.0, random_state=seed)
        n_test = int(len(cdf) * global_test_ratio)
        test_rows.append(cdf.iloc[:n_test])
        train_rows.append(cdf.iloc[n_test:])

    test_df = pd.concat(test_rows, ignore_index=True)
    train_df = pd.concat(train_rows, ignore_index=True)

    test_df[["image", "label"]].to_csv(os.path.join(output_dir, "global_test.csv"), index=False)
    print(f"[DEMOGRAPHIC PARTITION] ✅ Global test set saved: {len(test_df)} samples.")

    # Assign clients based on demographic criteria
    c1_mask = train_df["age_approx"] > 60
    c2_mask = train_df["age_approx"] <= 40
    c3_mask = (~c1_mask) & (~c2_mask) & (train_df["anatom_site_general"].isin(["head/neck", "upper extremity"]))
    c4_mask = (~c1_mask) & (~c2_mask) & (~c3_mask)

    client_dfs = {
        "client_1": train_df[c1_mask],
        "client_2": train_df[c2_mask],
        "client_3": train_df[c3_mask],
        "client_4": train_df[c4_mask],
    }

    for cid, cdf in client_dfs.items():
        save_p = os.path.join(output_dir, f"{cid}.csv")
        cdf[["image", "label"]].to_csv(save_p, index=False)
        print(f"[DEMOGRAPHIC PARTITION] ✅ {cid}: {len(cdf)} samples saved to {save_p}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="demographic", choices=["demographic", "class_bias"])
    args = parser.parse_args()

    if args.mode == "demographic":
        create_demographic_partitions()
    else:
        create_partitions()