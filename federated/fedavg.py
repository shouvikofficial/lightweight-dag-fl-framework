from federated.aggregator import Aggregator


# ============================================
# FEDAVG STRATEGY
# ============================================

class FedAvg:

    def __init__(self):

        self.aggregator = Aggregator()

    def aggregate(
        self,
        client_weights
    ):

        return self.aggregator.fedavg(
            client_weights
        )