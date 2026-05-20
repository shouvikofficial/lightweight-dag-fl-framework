from blockchain.hashing import verify_hash


class DAGValidator:
    """
    DAG transaction validation system.
    """

    def __init__(self, dag):

        self.dag = dag

    def validate_transaction_integrity(self, transaction):
        """
        Verify transaction hash integrity.
        """

        transaction_data = {
            "client_id": transaction.client_id,
            "model_hash": transaction.model_hash,
            "accuracy": transaction.accuracy,
            "timestamp": transaction.timestamp
        }

        return verify_hash(
            transaction_data,
            transaction.transaction_id
        )

    def validate_references(self, transaction):
        """
        Ensure referenced transactions exist.
        """

        for ref in transaction.references:

            if ref not in self.dag.nodes:
                return False

        return True

    def detect_duplicate_transaction(self, transaction):
        """
        Check if transaction already exists.
        """

        return transaction.transaction_id in self.dag.nodes

    def validate_transaction(self, transaction):
        """
        Full DAG validation pipeline.
        """

        # Duplicate check
        if self.detect_duplicate_transaction(transaction):

            print("[DAG] Duplicate transaction detected.")

            return False

        # Integrity check
        if not self.validate_transaction_integrity(transaction):

            print("[DAG] Transaction integrity failed.")

            return False

        # Reference validation
        if not self.validate_references(transaction):

            print("[DAG] Invalid DAG references.")

            return False

        return True