import random


# ============================================
# CLIENT SELECTION
# ============================================

class ClientSelector:

    def __init__(
        self,
        total_clients
    ):

        self.total_clients = total_clients

    # ========================================
    # RANDOM CLIENT SELECTION
    # ========================================

    def random_selection(
        self,
        fraction=1.0
    ):

        num_selected = max(
            1,
            int(self.total_clients * fraction)
        )

        selected_clients = random.sample(
            range(self.total_clients),
            num_selected
        )

        return selected_clients

    # ========================================
    # FIXED CLIENT SELECTION
    # ========================================

    def fixed_selection(self):

        return list(
            range(self.total_clients)
        )