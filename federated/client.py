# pyrefly: ignore [missing-import]
import flwr as fl
import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from datetime import datetime

from models.model import build_model, unfreeze_model, CategoricalFocalLoss
from preprocessing.balancing import get_class_weights
from blockchain.transaction import Transaction
from blockchain.hashing import generate_hash
from blockchain.shared_ledger import build_transaction


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
        fine_tune_round=2,
        fine_tune_at=120,
        fine_tune_lr=3e-5,
        mu=0.0,
        model_name="densenet121",
        attack_type="none",
        attack_factor=1.0,
    ):

        self.model = build_model(
            model_name=model_name,
            input_shape=(224, 224, 3),
            num_classes=5,
        )

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

        self.fine_tuned = False
        self.fine_tune_round = fine_tune_round
        self.fine_tune_at = fine_tune_at
        self.fine_tune_lr = fine_tune_lr
        self.mu = float(mu)
        self.attack_type = attack_type
        self.attack_factor = attack_factor

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
        import gc
        if self.y_test is None:
            y_true_batches = []
            y_prob_batches = []
            for i in range(len(self.x_test)):
                x_batch, y_batch = self.x_test[i]
                if isinstance(x_batch, (list, tuple)):
                    y_prob = self.model.predict([x_batch[0], x_batch[1]], verbose=0)
                else:
                    y_prob = self.model.predict(x_batch, verbose=0)
                y_true_batches.append(y_batch)
                y_prob_batches.append(y_prob)
                del x_batch
            y_true = np.concatenate(y_true_batches, axis=0)
            y_prob = np.concatenate(y_prob_batches, axis=0)
            del y_true_batches, y_prob_batches
            gc.collect()
            return y_true, y_prob

        if isinstance(self.x_test, (list, tuple)):
            y_prob = self.model.predict([self.x_test[0], self.x_test[1]], verbose=0)
        else:
            y_prob = self.model.predict(self.x_test, verbose=0)
        y_true = self.y_test
        gc.collect()
        return y_true, y_prob

    def _compute_extra_metrics(self, y_true, y_prob):
        metrics = {}
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_prob, axis=1)

        metrics["acc_manual"] = float(np.mean(y_true_labels == y_pred_labels))

        metrics["precision_macro"] = float(
            precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        )
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
        import gc
        gc.collect()

        self._log("Starting local training")
        self._log("Received global model from server")

        # Load global model weights
        self.model.set_weights(parameters)

        # ── FedProx: Inject true proximal regularization loss into optimizer ──
        if self.mu > 0.0:
            g_weights = [tf.constant(w, dtype=tf.float32) for w in parameters]
            model_ref = self.model
            mu_val = self.mu

            def fedprox_loss_fn(y_true, y_pred):
                focal = CategoricalFocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.02, num_classes=5)
                base_loss = focal(y_true, y_pred)
                prox_loss = 0.0
                for w, gw in zip(model_ref.trainable_weights, g_weights):
                    if w.shape == gw.shape:
                        prox_loss += tf.reduce_sum(tf.square(w - gw))
                return base_loss + (0.5 * mu_val * prox_loss)

            self.model.compile(
                optimizer=self.model.optimizer,
                loss=fedprox_loss_fn,
                metrics=["accuracy"],
            )

        # Train locally with validation data & progress bar
        if self.y_train is None:
            # ── Compute class weights from local generator data ──────────
            y_sample_batches = []
            sample_limit = min(len(self.x_train), 50)  # at most 50 batches
            for i in range(sample_limit):
                _, y_batch = self.x_train[i]
                y_sample_batches.append(y_batch)
            y_sample = np.concatenate(y_sample_batches, axis=0)
            try:
                class_weights = get_class_weights(y_sample, class_labels=list(range(5)), num_classes=5)
                self._log(
                    f"Class weights computed: "
                    + " | ".join(
                        f"cls{k}={v:.2f}" for k, v in class_weights.items()
                    )
                )
            except Exception as cw_err:
                self._log(f"[WARN] Class weights failed ({cw_err}), training unweighted")
                class_weights = None

            self.model.fit(
                self.x_train,
                epochs=6,
                steps_per_epoch=self.train_steps,
                validation_data=self.x_test,
                validation_steps=self.val_steps,
                class_weight=class_weights,
                verbose=1,
            )
            import gc
            gc.collect()
        else:
            y_train_arr = self.y_train
            try:
                class_weights = get_class_weights(y_train_arr, class_labels=list(range(5)), num_classes=5)
                self._log(
                    f"Class weights computed: "
                    + " | ".join(
                        f"cls{k}={v:.2f}" for k, v in class_weights.items()
                    )
                )
            except Exception as cw_err:
                self._log(f"[WARN] Class weights failed ({cw_err}), training unweighted")
                class_weights = None

            self.model.fit(
                self.x_train,
                self.y_train,
                epochs=6,
                batch_size=32,
                validation_data=(self.x_test, self.y_test),
                class_weight=class_weights,
                verbose=1,
            )
            import gc
            gc.collect()

        # Get updated weights
        updated_weights = self.model.get_weights()

        # Apply Attack Simulation if enabled
        if self.attack_type == "weight_noise":
            self._log(f"⚠️ [ATTACK] Injecting Gaussian weight noise (factor={self.attack_factor})")
            poisoned_weights = []
            for w in updated_weights:
                std = np.std(w) if np.std(w) > 0 else 0.1
                noise = np.random.normal(0, std * self.attack_factor, size=w.shape)
                poisoned_weights.append(w + noise)
            updated_weights = poisoned_weights
        elif self.attack_type == "label_flip":
            self._log("⚠️ [ATTACK] Executed Label Flipping attack during training")
            # Signify model perturbation for label flip attack
            poisoned_weights = []
            for w in updated_weights:
                std = np.std(w) if np.std(w) > 0 else 0.05
                noise = np.random.normal(0, std * 0.5 * self.attack_factor, size=w.shape)
                poisoned_weights.append(w + noise)
            updated_weights = poisoned_weights
        elif self.attack_type == "sign_flip":
            self._log(f"⚠️ [ATTACK] Executing Sign-Flipping Gradient Inversion (factor={self.attack_factor})")
            poisoned_weights = []
            for w, gw in zip(updated_weights, parameters):
                delta = w - gw
                poisoned_weights.append(gw - (self.attack_factor * delta))
            updated_weights = poisoned_weights
        elif self.attack_type == "free_rider":
            self._log("⚠️ [ATTACK] Executing Free-Rider Attack (zero computation, returning stale global weights)")
            updated_weights = [np.copy(p) for p in parameters]

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
        accuracy = metrics.get(
            "accuracy",
            metrics.get("categorical_accuracy", metrics.get("acc_manual", 0.0)),
        )

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
                "transaction": json.dumps(
                    build_transaction(
                        client_id=self.client_id,
                        round_number=int(config.get("round", 0)),
                        model_hash=model_hash,
                        accuracy=float(accuracy),
                        f1_macro=float(metrics.get("f1_macro", 0.0)),
                        roc_auc_ovr=float(metrics.get("roc_auc_ovr", 0.0)),
                    )
                ),
            },
        )

    # =========================
    # Global Evaluation
    # =========================

    def evaluate(self, parameters, config):
        import gc
        gc.collect()

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
        accuracy = metrics.get(
            "accuracy",
            metrics.get("categorical_accuracy", metrics.get("acc_manual", 0.0)),
        )
        metrics["accuracy"] = float(accuracy)

        del y_true, y_prob, eval_results
        gc.collect()

        if accuracy == 0.0:
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_prob, axis=1)
            acc_manual = float(np.mean(y_true_labels == y_pred_labels))
            true_counts = np.bincount(y_true_labels, minlength=y_prob.shape[1])
            pred_counts = np.bincount(y_pred_labels, minlength=y_prob.shape[1])
            self._log(
                "Eval debug | "
                f"acc_manual={acc_manual:.6f} | "
                f"metrics_names={self.model.metrics_names}"
            )
            self._log(
                "Eval debug | "
                f"eval_results={eval_results}"
            )
            self._log(
                "Eval debug | "
                f"true_counts={true_counts.tolist()} | "
                f"pred_counts={pred_counts.tolist()}"
            )

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