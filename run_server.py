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

import flwr as fl
from federated.aggregator import Aggregator
from federated.fedprox import FedProx


# ============================================
# CONFIG
# ============================================

SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 5
TOTAL_CLIENTS = 4


# ============================================
# CUSTOM FEDPROX STRATEGY
# ============================================

class FedProxStrategy(fl.server.strategy.FedAvg):

    def __init__(self, mu=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedprox = FedProx(mu=mu)
        self.aggregator = Aggregator()

    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        print(f"\n[Server] Round {server_round} — aggregating {len(results)} clients")

        client_weights = []
        client_sizes = []
        for _, fit_res in results:
            client_weights.append(
                fl.common.parameters_to_ndarrays(fit_res.parameters)
            )
            client_sizes.append(fit_res.num_examples)

        aggregated_weights = self.aggregator.weighted_average(
            client_weights,
            client_sizes,
        )

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)

        return aggregated_parameters, {}


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
    print(f"  Address  : {args.server}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Strategy : FedProx (mu=0.01)")
    print(f"  Total Clients: {args.total_clients}")
    print(f"  Min Clients: {min_fit_clients}")
    print(f"  Fraction Fit: {fraction_fit}")
    print("=" * 50)
    print("\n[Server] Waiting for clients...\n")

    fl.server.start_server(
        server_address=args.server,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    start_server(parse_args())
