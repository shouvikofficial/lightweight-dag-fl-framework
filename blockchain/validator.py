class Validator:
    """
    General federated transaction validator.
    """

    def __init__(self, min_accuracy=0.50):

        self.min_accuracy = min_accuracy

    def validate_accuracy(self, transaction):
        """
        Reject extremely poor model updates.
        """

        return transaction.accuracy >= self.min_accuracy

    def validate_model_hash(self, transaction):
        """
        Ensure model hash exists.
        """

        return transaction.model_hash is not None

    def validate_client(self, transaction):
        """
        Placeholder for trust score integration.
        """

        return True

    def validate(self, transaction):
        """
        Full validation pipeline.
        """

        if not self.validate_accuracy(transaction):

            print("[Validator] Accuracy validation failed.")

            return False

        if not self.validate_model_hash(transaction):

            print("[Validator] Invalid model hash.")

            return False

        if not self.validate_client(transaction):

            print("[Validator] Client validation failed.")

            return False

        return True