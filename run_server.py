"""
=====================================
 FEDERATED LEARNING SERVER LAUNCHER
=====================================

Run this FIRST in Terminal 1:
    python run_server.py

Then run clients in separate terminals.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from datetime import datetime
import time
import json
from typing import Tuple
import numpy as np

# pyrefly: ignore [missing-import]
# type: ignore
import flwr as fl
from blockchain.shared_ledger import add_transaction
from federated.aggregator import Aggregator
from federated.trust_aggregator import TrustAwareAggregator
from federated.fedprox import FedProx
from models.model import build_model
from models.evaluate import predict_batch_with_tta
from preprocessing.dataset_loader import prepare_global_test_generator
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


# ============================================
# CONFIG
# ============================================

SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 35
TOTAL_CLIENTS = 4
LOG_DIR = "logs"
SERVER_LOG_PATH = os.path.join(LOG_DIR, "server.log")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.jsonl")
GLOBAL_TEST_CSV = "dataset/partitions/global_test.csv"
IMAGE_ROOT = "dataset/raw/ISIC_2019_Training_Input"


def _trend_arrow(curr, prev, higher_is_better=True):
    if prev is None:
        return "-"
    if curr == prev:
        return "->"
    if higher_is_better:
        return "^" if curr > prev else "v"
    return "v" if curr < prev else "^"


def _fmt_time(seconds):
    if seconds is None:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _print_round_summary(
    server_round,
    clients,
    avg_loss,
    avg_metrics,
    prev_metrics,
    train_time_sec,
):
    accuracy = avg_metrics.get("accuracy", avg_metrics.get("acc_manual", 0.0))
    precision = avg_metrics.get("precision_macro", avg_metrics.get("precision", 0.0))
    recall = avg_metrics.get("recall_macro", avg_metrics.get("recall", 0.0))
    f1_macro = avg_metrics.get("f1_macro", 0.0)
    roc_auc = avg_metrics.get("roc_auc_ovr", 0.0)
    cm = avg_metrics.get("confusion_matrix")

    prev_loss = prev_metrics.get("loss") if prev_metrics else None
    prev_acc = prev_metrics.get("accuracy") if prev_metrics else None
    prev_f1 = prev_metrics.get("f1_macro") if prev_metrics else None
    prev_auc = prev_metrics.get("roc_auc_ovr") if prev_metrics else None

    print("=" * 64)
    print(f"📊 ROUND {server_round} — GLOBAL MEDICAL CLASSIFICATION EVALUATION")
    print("=" * 64)
    print(f"Participating Clients : {', '.join(clients) if clients else 'N/A'}")
    print(f"Training Time         : {_fmt_time(train_time_sec)}")
    print("-" * 64)
    print("Metric Summary (All 6 Medical Classification Metrics)")
    print("-" * 64)
    print(f"  1. Loss                  : {avg_loss:.4f} {_trend_arrow(avg_loss, prev_loss, False)}")
    print(f"  2. Accuracy              : {accuracy * 100:.2f}% {_trend_arrow(accuracy, prev_acc, True)}")
    print(f"  3. Precision (Macro)     : {precision:.4f}")
    print(f"  4. Recall / Sensitivity  : {recall:.4f}")
    print(f"  5. F1-Score (Macro)      : {f1_macro:.4f} {_trend_arrow(f1_macro, prev_f1, True)}")
    print(f"  6. ROC-AUC (Macro OVR)   : {roc_auc:.4f} {_trend_arrow(roc_auc, prev_auc, True)}")

    if cm is not None:
        print("-" * 64)
        print("  6. Confusion Matrix (8 ISIC 2019 Classes)")
        print("     Classes: MEL  NV  BKL  BCC   AK VASC   DF  SCC")
        class_names = ["MEL", "NV", "BKL", "BCC", "AK", "VASC", "DF", "SCC"]
        for idx, row in enumerate(cm):
            row_str = " ".join(f"{val:4d}" for val in row)
            cname = class_names[idx] if idx < len(class_names) else f"C{idx}"
            print(f"     {cname:<4} [ {row_str} ]")

    print("=" * 64 + "\n")


def _log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [SERVER] {message}"
    print(line)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SERVER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _evaluate_global_test(weights, model_name="densenet121", use_tta=True) -> dict:
    test_gen = prepare_global_test_generator(
        GLOBAL_TEST_CSV,
        IMAGE_ROOT,
        model_name=model_name,
    )

    num_classes = len(test_gen.class_indices)
    model = build_model(model_name=model_name, num_classes=num_classes, input_shape=(224, 224, 3))
    model.set_weights(weights)

    eval_results = model.evaluate(test_gen, verbose=0)
    loss = float(eval_results[0]) if isinstance(eval_results, (list, tuple)) else float(eval_results)

    y_true = []
    y_prob = []
    for i in range(len(test_gen)):
        x_batch, y_batch = test_gen[i]
        if use_tta:
            y_pred = predict_batch_with_tta(model, x_batch)
        else:
            if isinstance(x_batch, (list, tuple)):
                y_pred = model.predict([x_batch[0], x_batch[1]], verbose=0)
            else:
                y_pred = model.predict(x_batch, verbose=0)
        y_true.append(y_batch)
        y_prob.append(y_pred)

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_prob, axis=1)

    acc = float(np.mean(y_true_labels == y_pred_labels))
    prec = float(precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    rec = float(recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(range(num_classes)))

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "roc_auc_ovr": auc,
        "confusion_matrix": cm,
    }


# ============================================
# CUSTOM FEDPROX STRATEGY
# ============================================

class FedProxStrategy(fl.server.strategy.FedAvg):

    def __init__(self, mu=0.01, total_rounds=NUM_ROUNDS, model_name="densenet121", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedprox = FedProx(mu=mu)
        self.aggregator = Aggregator()
        self.trust_aggregator = TrustAwareAggregator(accept_threshold=0.80, reject_threshold=0.50)
        self.round_start_time = {}
        self.prev_metrics = None
        self.last_round_clients = []
        self.total_rounds = total_rounds
        self.model_name = model_name
        self._latest_weights = None
        self.trust_log = []

    def configure_fit(self, server_round, parameters, client_manager):
        fit_config = super().configure_fit(server_round, parameters, client_manager)
        self.round_start_time[server_round] = time.time()

        updated_config = []
        for client, fit_ins in fit_config:
            fit_ins.config["round"] = server_round
            updated_config.append((client, fit_ins))

        _log(
            f"Round {server_round} - selected {len(updated_config)} clients for training"
        )
        return updated_config

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_config = super().configure_evaluate(server_round, parameters, client_manager)
        _log(
            f"Round {server_round} - selected {len(eval_config)} clients for evaluation"
        )
        return eval_config

    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        _log(f"Round {server_round} - aggregating {len(results)} clients")

        client_weights = []
        client_sizes = []
        client_ids = []
        client_accuracies = {}
        blockchain_validations = {}

        for _, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")
            acc = fit_res.metrics.get("accuracy", fit_res.metrics.get("categorical_accuracy", 0.80))
            client_accuracies[client_id] = float(acc)
            is_valid = True

            tx_json = fit_res.metrics.get("transaction")
            if tx_json:
                try:
                    tx = json.loads(tx_json)
                    status = add_transaction(tx)
                    _log("[BLOCKCHAIN] Transaction verified")
                    if status.get("validated"):
                        _log("[BLOCKCHAIN] Update accepted")
                        is_valid = True
                    else:
                        _log("[BLOCKCHAIN] Update rejected")
                        is_valid = False
                except json.JSONDecodeError:
                    _log("[BLOCKCHAIN] Transaction parse failed")
                    is_valid = False

            blockchain_validations[client_id] = is_valid
            client_weights.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
            client_sizes.append(fit_res.num_examples)
            client_ids.append(client_id)

        _log("Received updates from: " + ", ".join(client_ids))

        # Perform 4-Factor Adaptive Trust-Weighted Aggregation with 3-Tier Decision System
        aggregated_weights, trust_info = self.trust_aggregator.aggregate(
            client_ids=client_ids,
            client_weights=client_weights,
            client_sizes=client_sizes,
            client_accuracies=client_accuracies,
            blockchain_validations=blockchain_validations,
            prev_global_weights=self._latest_weights,
        )

        # Print 3-Tier Trust & Security Evaluation Table in Server Terminal
        print("\n" + "🛡️  " + "═"*72)
        print(f"   ROUND {server_round} · 4-FACTOR ADAPTIVE TRUST & SECURITY ASSESSMENT")
        print("   " + "─"*72)
        print(f"   {'Client ID':<12} {'Similarity':<12} {'Val Acc':<10} {'Trust Score':<14} {'3-Tier Decision':<20}")
        print("   " + "─"*72)
        for cid, info in trust_info.items():
            act = info["action"]
            if act == "ACCEPT":
                act_str = "✅ ACCEPT (100% Weight)"
            elif act == "PENALIZE":
                act_str = "⚠️ PENALIZE (50% Weight)"
            else:
                act_str = "🚫 REJECT (0% Excluded)"
            print(f"   {cid:<12} {info['similarity']:<12.4f} {info['val_accuracy']:<10.4f} {info['trust_score']:<14.4f} {act_str:<20}")
        print("   " + "─"*72)
        print("   Formula: Trust = 0.35*Hist + 0.25*Sim + 0.20*Acc + 0.20*Blockchain")
        print("   Tiers:   ≥0.80 -> ACCEPT  |  0.50-0.79 -> PENALIZE  |  <0.50 -> REJECT")
        print("🛡️  " + "═"*72 + "\n")

        # Record trust scores to file
        round_trust_entry = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "client_trust": trust_info
        }
        self.trust_log.append(round_trust_entry)
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "trust_scores.json"), "w", encoding="utf-8") as tf_file:
                json.dump(self.trust_log, tf_file, indent=2)
        except Exception as log_err:
            _log(f"[WARN] Failed writing trust_scores.json: {log_err}")

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

        self._latest_weights = aggregated_weights
        self.last_round_clients = client_ids
        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):

        if not results:
            return None, {}

        total_examples = 0
        weighted_loss = 0.0
        weighted_metrics = {}

        for _, eval_res in results:
            num_examples = eval_res.num_examples
            total_examples += num_examples
            weighted_loss += eval_res.loss * num_examples

            for key, value in eval_res.metrics.items():
                weighted_metrics[key] = (
                    weighted_metrics.get(key, 0.0) + value * num_examples
                )

        avg_loss = weighted_loss / total_examples if total_examples else None
        avg_metrics = {
            key: value / total_examples for key, value in weighted_metrics.items()
        }

        if avg_loss is not None:
            metrics_str = " | ".join(
                f"{k}={v:.4f}" for k, v in avg_metrics.items()
            )
            _log(
                f"Round {server_round} eval: loss={avg_loss:.4f}"
                + (f" | {metrics_str}" if metrics_str else "")
            )

            os.makedirs(LOG_DIR, exist_ok=True)
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "round": server_round,
                "loss": avg_loss,
                "metrics": avg_metrics,
            }
            with open(METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

            train_time_sec = None
            if server_round in self.round_start_time:
                train_time_sec = time.time() - self.round_start_time[server_round]

            _print_round_summary(
                server_round,
                self.last_round_clients,
                avg_loss,
                avg_metrics,
                self.prev_metrics,
                train_time_sec,
            )
            self.prev_metrics = {
                "loss": avg_loss,
                "accuracy": avg_metrics.get("accuracy", 0.0),
                "f1_macro": avg_metrics.get("f1_macro", 0.0),
                "roc_auc_ovr": avg_metrics.get("roc_auc_ovr", 0.0),
            }

            if server_round == self.total_rounds and self._latest_weights is not None:
                _log("Final global test evaluation starting")
                try:
                    eval_res = _evaluate_global_test(self._latest_weights, model_name=self.model_name)
                    _log(
                        "Final global test | "
                        f"loss={eval_res['loss']:.4f} | acc={eval_res['accuracy']:.4f} | "
                        f"prec={eval_res['precision']:.4f} | rec={eval_res['recall']:.4f} | "
                        f"f1={eval_res['f1_macro']:.4f} | auc={eval_res['roc_auc_ovr']:.4f}"
                    )
                except Exception as exc:
                    _log(f"Global test evaluation failed: {exc}")

        return avg_loss, avg_metrics


# ============================================
# START SERVER
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the Federated Learning server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=SERVER_ADDRESS,
        help=f"Server address (default: {SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of FL rounds (default: {NUM_ROUNDS})",
    )
    parser.add_argument(
        "--total_clients",
        type=int,
        default=TOTAL_CLIENTS,
        help=f"Total number of clients (default: {TOTAL_CLIENTS})",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=None,
        help="Min clients per round (default: wait for all)",
    )
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=None,
        help="Fraction of clients per round (default: 1.0)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet201",
        choices=["densenet121", "densenet169", "densenet201", "resnet50v2", "efficientnetb0"],
        help="Backbone architecture to use (default: densenet201)",
    )
    return parser.parse_args()


def start_server(args):

    os.makedirs(LOG_DIR, exist_ok=True)

    min_fit_clients = args.min_fit_clients or args.total_clients
    min_evaluate_clients = min_fit_clients
    min_available_clients = args.total_clients
    fraction_fit = args.fraction_fit if args.fraction_fit is not None else 1.0

    strategy = FedProxStrategy(
        mu=0.01,
        total_rounds=args.rounds,
        model_name=args.model_name,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )

    print("=" * 50)
    print("   FEDERATED LEARNING SERVER")
    print("=" * 50)
    _log(f"Address: {args.server}")
    _log(f"Rounds: {args.rounds}")
    _log("Strategy: FedProx (mu=0.01)")
    _log(f"Total clients: {args.total_clients}")
    _log(f"Min clients per round: {min_fit_clients}")
    _log(f"Fraction fit: {fraction_fit}")
    print("=" * 50)
    _log("Waiting for clients...")

    fl.server.start_server(
        server_address=args.server,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    start_server(parse_args())
