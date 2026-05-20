import numpy as np


# ============================================
# BANDWIDTH MONITOR
# ============================================

class BandwidthMonitor:

    def __init__(self):

        self.total_bytes_sent = 0

    # ========================================
    # TRACK MODEL SIZE
    # ========================================

    def calculate_size(
        self,
        weights
    ):

        total_size = 0

        for layer in weights:

            total_size += layer.nbytes

        return total_size

    # ========================================
    # UPDATE TOTAL BANDWIDTH
    # ========================================

    def update_usage(
        self,
        weights
    ):

        size = self.calculate_size(weights)

        self.total_bytes_sent += size

        return size

    # ========================================
    # GET TOTAL BANDWIDTH
    # ========================================

    def get_total_usage(self):

        return self.total_bytes_sent