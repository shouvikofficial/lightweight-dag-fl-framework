import time


# ============================================
# LATENCY TRACKER
# ============================================

class LatencyTracker:

    def __init__(self):

        self.start_time = None

    # ========================================
    # START TIMER
    # ========================================

    def start(self):

        self.start_time = time.time()

    # ========================================
    # STOP TIMER
    # ========================================

    def stop(self):

        if self.start_time is None:

            return None

        latency = (
            time.time() - self.start_time
        )

        self.start_time = None

        return latency