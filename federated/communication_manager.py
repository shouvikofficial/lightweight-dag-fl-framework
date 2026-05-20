import time


# ============================================
# COMMUNICATION MANAGER
# ============================================

class CommunicationManager:

    def __init__(self):

        self.total_upload_time = 0
        self.total_download_time = 0

    # ========================================
    # TRACK UPLOAD LATENCY
    # ========================================

    def track_upload(self):

        start_time = time.time()

        return start_time

    def end_upload(self, start_time):

        elapsed = time.time() - start_time

        self.total_upload_time += elapsed

        return elapsed

    # ========================================
    # TRACK DOWNLOAD LATENCY
    # ========================================

    def track_download(self):

        start_time = time.time()

        return start_time

    def end_download(self, start_time):

        elapsed = time.time() - start_time

        self.total_download_time += elapsed

        return elapsed

    # ========================================
    # GET TOTAL STATS
    # ========================================

    def get_statistics(self):

        return {
            "upload_time":
                self.total_upload_time,

            "download_time":
                self.total_download_time
        }