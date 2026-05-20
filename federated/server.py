import flwr as fl

from federated.fedprox import FedProx


# =========================
# Custom FedProx Strategy
# =========================

class FedProxStrategy(fl.server.strategy.FedAvg):

    def __init__(self, mu=0.01, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.fedprox = FedProx(mu=mu)

    # =========================
    # Aggregate Client Updates
    # =========================

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        if not results:
            return None, {}

        # Extract weights
        client_weights = [
            fl.common.parameters_to_ndarrays(
                fit_res.parameters
            )
            for _, fit_res in results
        ]

        # FedProx aggregation
        aggregated_weights = self.fedprox.aggregate(
            client_weights
        )

        # Convert back to Flower format
        aggregated_parameters = (
            fl.common.ndarrays_to_parameters(
                aggregated_weights
            )
        )

        return aggregated_parameters, {}


# =========================
# Start Flower Server
# =========================

def start_server():

    strategy = FedProxStrategy(
        mu=0.01,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(
            num_rounds=5
        )
    )


if __name__ == "__main__":

    start_server()