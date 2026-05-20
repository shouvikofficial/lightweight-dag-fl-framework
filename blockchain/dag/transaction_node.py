class TransactionNode:
    """
    Represents a node inside the DAG structure.
    """

    def __init__(self, transaction):

        self.transaction = transaction

        # Parent transaction references
        self.references = transaction.references

        # Validation status
        self.validated = False

    def validate(self):
        """
        Mark transaction node as validated.
        """

        self.validated = True

    def to_dict(self):
        """
        Convert node into dictionary format.
        """

        return {
            "transaction_id": self.transaction.transaction_id,
            "client_id": self.transaction.client_id,
            "model_hash": self.transaction.model_hash,
            "accuracy": self.transaction.accuracy,
            "references": self.references,
            "timestamp": self.transaction.timestamp,
            "validated": self.validated
        }

    def __repr__(self):

        return (
            f"TransactionNode("
            f"tx_id={self.transaction.transaction_id[:10]}, "
            f"validated={self.validated})"
        )