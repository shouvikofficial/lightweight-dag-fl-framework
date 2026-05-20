import json

from blockchain.dag.transaction_node import TransactionNode


class DAG:
    """
    Lightweight DAG structure for federated learning transactions.
    """

    def __init__(self):

        # Store all DAG nodes
        self.nodes = {}

    def add_transaction(self, transaction):
        """
        Add new transaction into DAG.
        """

        node = TransactionNode(transaction)

        # Add node using transaction ID
        self.nodes[transaction.transaction_id] = node

        return node

    def validate_transaction(self, transaction_id):
        """
        Validate a DAG transaction node.
        """

        if transaction_id in self.nodes:

            self.nodes[transaction_id].validate()

            return True

        return False

    def get_transaction(self, transaction_id):
        """
        Retrieve transaction node by ID.
        """

        return self.nodes.get(transaction_id)

    def get_all_transactions(self):
        """
        Return all DAG transactions.
        """

        return list(self.nodes.values())

    def get_tips(self):
        """
        DAG tips:
        transactions that are NOT referenced by others.
        """

        referenced_transactions = set()

        # Collect all referenced transaction IDs
        for node in self.nodes.values():

            for ref in node.references:
                referenced_transactions.add(ref)

        # Tips are unreferenced nodes
        tips = []

        for tx_id, node in self.nodes.items():

            if tx_id not in referenced_transactions:
                tips.append(node)

        return tips

    def save_ledger(self, filepath="blockchain/ledger.json"):
        """
        Save DAG ledger into JSON file.
        """

        ledger_data = []

        for node in self.nodes.values():
            ledger_data.append(node.to_dict())

        with open(filepath, "w") as file:
            json.dump(ledger_data, file, indent=4)

    def load_ledger(self, filepath="blockchain/ledger.json"):
        """
        Load ledger data.
        """

        try:
            with open(filepath, "r") as file:
                return json.load(file)

        except FileNotFoundError:
            return []

    def __repr__(self):

        return f"DAG(total_transactions={len(self.nodes)})"