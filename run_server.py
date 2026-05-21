"""
=====================================
 FEDERATED LEARNING SERVER LAUNCHER
=====================================

Run this FIRST in Terminal 1:
    python run_server.py

Then run clients in separate terminals.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from datetime import datetime
import time
import json

import flwr as fl
from blockchain.shared_ledger import add_transaction
from federated.aggregator import Aggregator
from federated.fedprox import FedProx


# ============================================
# CONFIG
# ============================================

SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 20
TOTAL_CLIENTS = 4
LOG_DIR = "logs"
SERVER_LOG_PATH = os.path.join(LOG_DIR, "server.log")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.jsonl")


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
    accuracy = avg_metrics.get("accuracy", 0.0)
    f1_macro = avg_metrics.get("f1_macro", 0.0)
    roc_auc = avg_metrics.get("roc_auc_ovr", 0.0)

    prev_loss = prev_metrics.get("loss") if prev_metrics else None
    prev_acc = prev_metrics.get("accuracy") if prev_metrics else None
    prev_f1 = prev_metrics.get("f1_macro") if prev_metrics else None
    prev_auc = prev_metrics.get("roc_auc_ovr") if prev_metrics else None

    print("=" * 50)
    print(f"ROUND {server_round} - GLOBAL EVALUATION")
    print("=" * 50)
    print()
    print(f"Participating Clients : {', '.join(clients) if clients else 'N/A'}")
    print(f"Training Time         : {_fmt_time(train_time_sec)}")
    print()
    print("Metric Summary")
    print("-" * 14)
    print(f"Loss        : {avg_loss:.4f} {_trend_arrow(avg_loss, prev_loss, False)}")
    print(f"Accuracy    : {accuracy * 100:.2f}% {_trend_arrow(accuracy, prev_acc, True)}")
    print(f"F1 Macro    : {f1_macro:.4f} {_trend_arrow(f1_macro, prev_f1, True)}")
    print(f"ROC AUC OVR : {roc_auc:.4f} {_trend_arrow(roc_auc, prev_auc, True)}")
    print()
    print("Training Status")
    print("-" * 15)
    status = []
    if prev_loss is not None and avg_loss < prev_loss:
        status.append("Global model converging successfully")
    if prev_loss is not None:
        status.append("Loss decreasing steadily" if avg_loss < prev_loss else "Loss is not improving")
    if prev_auc is not None:
        status.append("AUC improving consistently" if roc_auc > prev_auc else "AUC plateauing")
    if prev_f1 is not None:
        status.append("Minority-class prediction improving" if f1_macro > prev_f1 else "F1 needs more rounds")
    if not status:
        status.append("Waiting for next round to assess trends")
    for item in status:
        print(f"* {item}")
    print()
    print("=" * 50)


def _log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [SERVER] {message}"
    print(line)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SERVER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================
# CUSTOM FEDPROX STRATEGY
# ============================================

class FedProxStrategy(fl.server.strategy.FedAvg):

    def __init__(self, mu=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedprox = FedProx(mu=mu)
        self.aggregator = Aggregator()
        self.round_start_time = {}
        self.prev_metrics = None
        self.last_round_clients = []

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
        for _, fit_res in results:
            client_weights.append(
                fl.common.parameters_to_ndarrays(fit_res.parameters)
            )
            client_sizes.append(fit_res.num_examples)
            client_ids.append(fit_res.metrics.get("client_id", "unknown"))

            tx_json = fit_res.metrics.get("transaction")
            if tx_json:
                try:
                    tx = json.loads(tx_json)
                    status = add_transaction(tx)
                    _log("[BLOCKCHAIN] Transaction verified")
                    if status.get("reason") == "bad_signature":
                        _log("[BLOCKCHAIN] Signature invalid")
                    else:
                        _log("[BLOCKCHAIN] Signature valid")

                    if status.get("validated"):
                        _log("[BLOCKCHAIN] Update accepted")
                    else:
                        _log("[BLOCKCHAIN] Update rejected")
                except json.JSONDecodeError:
                    _log("[BLOCKCHAIN] Transaction parse failed")

        _log(
            "Received updates from: "
            + ", ".join(client_ids)
        )

        aggregated_weights = self.aggregator.weighted_average(
            client_weights,
            client_sizes,
        )

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

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
    return parser.parse_args()


def start_server(args):

    os.makedirs(LOG_DIR, exist_ok=True)

    min_fit_clients = args.min_fit_clients or args.total_clients
    min_evaluate_clients = min_fit_clients
    min_available_clients = args.total_clients
    fraction_fit = args.fraction_fit if args.fraction_fit is not None else 1.0

    strategy = FedProxStrategy(
        mu=0.01,
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
