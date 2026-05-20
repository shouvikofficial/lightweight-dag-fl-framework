from blockchain.transaction import Transaction
from blockchain.dag.dag_structure import DAG
from blockchain.dag.dag_validator import DAGValidator
from blockchain.dag.tip_selection import TipSelector
from blockchain.hashing import generate_hash


# =========================
# Initialize DAG
# =========================

dag = DAG()

validator = DAGValidator(dag)

tip_selector = TipSelector(dag)


# =========================
# Create First Transaction
# =========================

model_update_1 = generate_hash("client_1_model_weights")

tx1 = Transaction(
    client_id="client_1",
    model_hash=model_update_1,
    accuracy=0.91
)

# Validate transaction
if validator.validate_transaction(tx1):

    node1 = dag.add_transaction(tx1)

    dag.validate_transaction(tx1.transaction_id)

    print("[+] Transaction 1 Added")


# =========================
# Create Second Transaction
# =========================

tips = tip_selector.random_tip_selection()

model_update_2 = generate_hash("client_2_model_weights")

tx2 = Transaction(
    client_id="client_2",
    model_hash=model_update_2,
    accuracy=0.94,
    references=tips
)

if validator.validate_transaction(tx2):

    node2 = dag.add_transaction(tx2)

    dag.validate_transaction(tx2.transaction_id)

    print("[+] Transaction 2 Added")


# =========================
# Create Third Transaction
# =========================

tips = tip_selector.highest_accuracy_tip_selection()

model_update_3 = generate_hash("client_3_model_weights")

tx3 = Transaction(
    client_id="client_3",
    model_hash=model_update_3,
    accuracy=0.96,
    references=tips
)

if validator.validate_transaction(tx3):

    node3 = dag.add_transaction(tx3)

    dag.validate_transaction(tx3.transaction_id)

    print("[+] Transaction 3 Added")


# =========================
# Save DAG Ledger
# =========================

dag.save_ledger()

print("\n===== DAG Transactions =====")

for node in dag.get_all_transactions():

    print(node.to_dict())


# =========================
# Show DAG Tips
# =========================

print("\n===== DAG Tips =====")

tips = dag.get_tips()

for tip in tips:

    print(tip.transaction.transaction_id[:10])