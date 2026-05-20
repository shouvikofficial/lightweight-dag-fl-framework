import numpy as np


class FedProx:
    """
    FedProx aggregation strategy for federated learning.
    """

    def __init__(self, mu=0.01):

        # Proximal term coefficient
        self.mu = mu

    def aggregate(self, client_weights):
        """
        Aggregate client model weights.

        Parameters:
        -----------
        client_weights : list
            List of model weights from clients.

        Returns:
        --------
        aggregated_weights : list
        """

        if len(client_weights) == 0:
            return None

        aggregated_weights = []

        # Iterate layer-wise
        for layer_weights in zip(*client_weights):

            layer_average = np.mean(
                np.array(layer_weights),
                axis=0
            )

            aggregated_weights.append(layer_average)

        return aggregated_weights

    def proximal_term(
        self,
        local_weights,
        global_weights
    ):
        """
        Compute FedProx proximal regularization term.

        || w_local - w_global ||²
        """

        proximal_loss = 0.0

        for lw, gw in zip(local_weights, global_weights):

            proximal_loss += np.linalg.norm(lw - gw) ** 2

        return (self.mu / 2) * proximal_loss