import flwr as fl
import tensorflow as tf
import numpy as np

from models.model import build_model
from blockchain.transaction import Transaction
from blockchain.hashing import generate_hash


class FLClient(fl.client.NumPyClient):

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        client_id,
        dag,
        validator,
        train_steps=None,
        val_steps=None,
        train_samples=None,
        val_samples=None,
    ):

        self.model = build_model()

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.train_steps = train_steps
        self.val_steps = val_steps
        self.train_samples = train_samples
        self.val_samples = val_samples

        self.client_id = client_id

        self.dag = dag
        self.validator = validator

    # =========================
    # Get Model Parameters
    # =========================

    def get_parameters(self, config):

        return self.model.get_weights()

    # =========================
    # Local Training
    # =========================

    def fit(self, parameters, config):

        # Load global model weights
        self.model.set_weights(parameters)

        # Train locally
        if self.y_train is None:
            self.model.fit(
                self.x_train,
                epochs=1,
                steps_per_epoch=self.train_steps,
                verbose=1,
            )
        else:
            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=1,
                batch_size=32,
                verbose=1,
            )

        # Get updated weights
        updated_weights = self.model.get_weights()

        # Evaluate local accuracy
        if self.y_test is None:
            eval_results = self.model.evaluate(
                self.x_test,
                steps=self.val_steps,
                verbose=0,
            )
        else:
            eval_results = self.model.evaluate(
                self.x_test,
                self.y_test,
                verbose=0,
            )

        if isinstance(eval_results, (list, tuple)):
            loss = float(eval_results[0])
            accuracy = float(eval_results[1]) if len(eval_results) > 1 else 0.0
        else:
            loss = float(eval_results)
            accuracy = 0.0

        # =========================
        # Generate Model Hash
        # =========================

        model_hash = generate_hash(
            str(updated_weights[0].tolist())
        )

        # =========================
        # Create DAG Transaction
        # =========================

        transaction = Transaction(
            client_id=self.client_id,
            model_hash=model_hash,
            accuracy=float(accuracy)
        )

        # Validate transaction
        if self.validator.validate_transaction(transaction):

            self.dag.add_transaction(transaction)

            self.dag.validate_transaction(
                transaction.transaction_id
            )

            print(
                f"[DAG] Transaction Added "
                f"from {self.client_id}"
            )

        num_examples = (
            self.train_samples
            if self.train_samples is not None
            else len(self.x_train)
        )

        return (
            updated_weights,
            num_examples,
            {"accuracy": float(accuracy)}
        )

    # =========================
    # Global Evaluation
    # =========================

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)

        if self.y_test is None:
            eval_results = self.model.evaluate(
                self.x_test,
                steps=self.val_steps,
                verbose=0,
            )
        else:
            eval_results = self.model.evaluate(
                self.x_test,
                self.y_test,
                verbose=0
            )

        if isinstance(eval_results, (list, tuple)):
            loss = float(eval_results[0])
            accuracy = float(eval_results[1]) if len(eval_results) > 1 else 0.0
        else:
            loss = float(eval_results)
            accuracy = 0.0

        num_examples = (
            self.val_samples
            if self.val_samples is not None
            else len(self.x_test)
        )

        return loss, num_examples, {
            "accuracy": float(accuracy)
        }