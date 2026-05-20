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
import json

import flwr as fl
from federated.aggregator import Aggregator
from federated.fedprox import FedProx


# ============================================
# CONFIG
# ============================================

SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 5
TOTAL_CLIENTS = 4
LOG_DIR = "logs"
SERVER_LOG_PATH = os.path.join(LOG_DIR, "server.log")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.jsonl")


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

    def configure_fit(self, server_round, parameters, client_manager):
        fit_config = super().configure_fit(server_round, parameters, client_manager)
        client_ids = [client.cid for client, _ in fit_config]
        _log(
            f"Round {server_round} - selected clients for training: "
            f"{', '.join(client_ids)}"
        )
        return fit_config

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_config = super().configure_evaluate(server_round, parameters, client_manager)
        client_ids = [client.cid for client, _ in eval_config]
        _log(
            f"Round {server_round} - selected clients for evaluation: "
            f"{', '.join(client_ids)}"
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

        _log(
            "Received updates from: "
            + ", ".join(
                f"{cid} ({size} samples)"
                for cid, size in zip(client_ids, client_sizes)
            )
        )

        aggregated_weights = self.aggregator.weighted_average(
            client_weights,
            client_sizes,
        )

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

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
