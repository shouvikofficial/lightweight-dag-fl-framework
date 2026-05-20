import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime

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
        log_path=None,
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
        self.log_path = log_path

    # =========================
    # Get Model Parameters
    # =========================

    def get_parameters(self, config):

        return self.model.get_weights()

    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{self.client_id.upper()}] {message}"
        print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _extract_metrics(self, eval_results):
        if not isinstance(eval_results, (list, tuple)):
            return float(eval_results), {}

        metrics = {}
        metric_names = list(self.model.metrics_names)
        loss = float(eval_results[0])

        for name, value in zip(metric_names[1:], eval_results[1:]):
            metrics[name] = float(value)

        return loss, metrics

    def _collect_eval_data(self):
        if self.y_test is None:
            y_true_batches = []
            y_prob_batches = []
            for i in range(len(self.x_test)):
                x_batch, y_batch = self.x_test[i]
                y_prob = self.model.predict(x_batch, verbose=0)
                y_true_batches.append(y_batch)
                y_prob_batches.append(y_prob)
            y_true = np.concatenate(y_true_batches, axis=0)
            y_prob = np.concatenate(y_prob_batches, axis=0)
            return y_true, y_prob

        y_true = self.y_test
        y_prob = self.model.predict(self.x_test, verbose=0)
        return y_true, y_prob

    def _compute_extra_metrics(self, y_true, y_prob):
        metrics = {}
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_prob, axis=1)

        metrics["f1_macro"] = float(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        )

        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr"] = 0.0

        return metrics

    # =========================
    # Local Training
    # =========================

    def fit(self, parameters, config):

        self._log("Starting local training")
        self._log("Received global model from server")

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

        loss, metrics = self._extract_metrics(eval_results)

        y_true, y_prob = self._collect_eval_data()
        metrics.update(self._compute_extra_metrics(y_true, y_prob))
        accuracy = metrics.get("accuracy", 0.0)

        self._log(
            "Local training done | "
            f"val_loss={loss:.4f} | val_acc={accuracy:.4f} | "
            f"val_f1={metrics.get('f1_macro', 0.0):.4f} | "
            f"val_auc={metrics.get('roc_auc_ovr', 0.0):.4f}"
        )

        self._log("Sending updated model to server")

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
            {
                **(metrics if metrics else {"accuracy": float(accuracy)}),
                "client_id": self.client_id,
            },
        )

    # =========================
    # Global Evaluation
    # =========================

    def evaluate(self, parameters, config):

        self._log("Starting evaluation")

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

        loss, metrics = self._extract_metrics(eval_results)

        y_true, y_prob = self._collect_eval_data()
        metrics.update(self._compute_extra_metrics(y_true, y_prob))
        accuracy = metrics.get("accuracy", 0.0)

        num_examples = (
            self.val_samples
            if self.val_samples is not None
            else len(self.x_test)
        )

        self._log(
            "Evaluation done | "
            f"loss={loss:.4f} | acc={accuracy:.4f} | "
            f"f1={metrics.get('f1_macro', 0.0):.4f} | "
            f"auc={metrics.get('roc_auc_ovr', 0.0):.4f}"
        )

        return loss, num_examples, metrics if metrics else {
            "accuracy": float(accuracy)
        }