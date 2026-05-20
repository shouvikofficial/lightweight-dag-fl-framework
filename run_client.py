"""
=====================================
 FEDERATED LEARNING CLIENT LAUNCHER
=====================================

Run this AFTER starting the server.
Open a separate terminal for EACH client:

    python run_client.py --client_id client_1
    python run_client.py --client_id client_2
    python run_client.py --client_id client_3
    python run_client.py --client_id client_4

Available client IDs: client_1, client_2, client_3, client_4
"""

import os
import sys
import argparse
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl

from federated.client import FLClient
from preprocessing.dataset_loader import (
    prepare_client_data,
    prepare_client_generators,
)
from blockchain.dag.dag_structure import DAG
from blockchain.dag.dag_validator import DAGValidator


# ============================================
# CONFIG
# ============================================

SERVER_ADDRESS = "localhost:8080"

DATASET_DIR    = "dataset/partitions"
IMAGE_ROOT     = "dataset/raw/ISIC_2019_Training_Input"

VALID_CLIENTS  = ["client_1", "client_2", "client_3", "client_4"]
LOG_DIR = "logs"


def _log(client_id, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{client_id.upper()}] {message}"
    print(line)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{client_id}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================
# ARGUMENT PARSING
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a Federated Learning client"
    )
    parser.add_argument(
        "--client_id",
        type=str,
        required=True,
        choices=VALID_CLIENTS,
        help="Client ID (e.g. client_1)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=SERVER_ADDRESS,
        help=f"Server address (default: {SERVER_ADDRESS})"
    )
    return parser.parse_args()


# ============================================
# START CLIENT
# ============================================

def start_client(client_id, server_address):

    csv_path = os.path.join(DATASET_DIR, f"{client_id}.csv")

    # ========================================
    # VALIDATE DATASET EXISTS
    # ========================================

    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Dataset not found: {csv_path}")
        print(f"\nPlease run the partitioning script first:")
        print(f"    python preprocessing/partition.py")
        sys.exit(1)

    if not os.path.exists(IMAGE_ROOT):
        print(f"\n[ERROR] Image directory not found: {IMAGE_ROOT}")
        print(f"\nDownload ISIC 2019 dataset and place images in:")
        print(f"    {IMAGE_ROOT}")
        sys.exit(1)

    # ========================================
    # LOAD CLIENT DATA
    # ========================================

    print("=" * 50)
    print(f"   CLIENT: {client_id.upper()}")
    print("=" * 50)
    _log(client_id, f"CSV: {csv_path}")
    _log(client_id, f"Images: {IMAGE_ROOT}")
    _log(client_id, f"Server: {server_address}")
    print("=" * 50)
    _log(client_id, "Loading dataset...")

    train_gen, val_gen, class_names = prepare_client_generators(
        csv_path=csv_path,
        image_root=IMAGE_ROOT,
    )

    train_samples = getattr(train_gen, "n", None) or getattr(train_gen, "samples", None)
    val_samples = getattr(val_gen, "n", None) or getattr(val_gen, "samples", None)

    _log(
        client_id,
        f"Train samples: {train_samples} | Val samples: {val_samples}",
    )

    # ========================================
    # SETUP BLOCKCHAIN
    # ========================================

    dag = DAG()
    validator = DAGValidator(dag)

    # ========================================
    # CREATE FLOWER CLIENT
    # ========================================

    client = FLClient(
        x_train=train_gen,
        y_train=None,
        x_test=val_gen,
        y_test=None,
        client_id=client_id,
        dag=dag,
        validator=validator,
        train_steps=len(train_gen),
        val_steps=len(val_gen),
        train_samples=train_samples,
        val_samples=val_samples,
        log_path=os.path.join(LOG_DIR, f"{client_id}.log"),
    )

    _log(client_id, f"Connecting to server at {server_address}...")

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )

    # ========================================
    # SAVE CLIENT DAG LEDGER
    # ========================================

    ledger_path = f"blockchain/ledger_{client_id}.json"
    dag.save_ledger(filepath=ledger_path)
    _log(client_id, f"DAG ledger saved -> {ledger_path}")


if __name__ == "__main__":
    args = parse_args()
    start_client(args.client_id, args.server)
