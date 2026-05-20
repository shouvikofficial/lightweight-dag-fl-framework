from datetime import datetime
from blockchain.hashing import generate_hash


class Transaction:
    """
    Represents a federated learning model update transaction.
    """

    def __init__(
        self,
        client_id,
        model_hash,
        accuracy,
        references=None
    ):

        self.client_id = client_id
        self.model_hash = model_hash
        self.accuracy = accuracy

        # DAG references to previous transactions
        self.references = references if references else []

        self.timestamp = str(datetime.utcnow())

        # Generate transaction ID
        self.transaction_id = self.generate_transaction_id()

    def to_dict(self):
        """
        Convert transaction object into dictionary.
        """

        return {
            "transaction_id": self.transaction_id,
            "client_id": self.client_id,
            "model_hash": self.model_hash,
            "accuracy": self.accuracy,
            "references": self.references,
            "timestamp": self.timestamp
        }

    def generate_transaction_id(self):
        """
        Generate unique transaction hash ID.
        """

        transaction_data = {
            "client_id": self.client_id,
            "model_hash": self.model_hash,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp
        }

        return generate_hash(transaction_data)

    def __repr__(self):

        return (
            f"Transaction("
            f"client_id={self.client_id}, "
            f"accuracy={self.accuracy})"
        )