import hmac
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from blockchain.hashing import generate_hash


LEDGER_PATH = "blockchain/ledger.json"


def _load_env_file(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}

    values = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_client_secrets() -> Dict[str, str]:
    env_values = _load_env_file(".env")

    secrets = {}
    for client_id in ["client_1", "client_2", "client_3", "client_4"]:
        env_key = f"{client_id.upper()}_HMAC_SECRET"
        secrets[client_id] = os.getenv(env_key, env_values.get(env_key, ""))
    return secrets


# Store secrets securely in production (env vars or vault).
CLIENT_SECRETS = _load_client_secrets()

CONSENSUS_ROC_AUC_MIN = 0.80


# =========================
# LEDGER IO
# =========================

def load_ledger(path: str = LEDGER_PATH) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def save_ledger(ledger: List[Dict], path: str = LEDGER_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=4)


# =========================
# HASHING + SIGNATURES
# =========================

def _signing_payload(tx: Dict) -> str:
    payload = {k: v for k, v in tx.items() if k != "signature"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_transaction(tx: Dict, secret: str) -> str:
    payload = _signing_payload(tx).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_signature(tx: Dict, secret: str) -> bool:
    if "signature" not in tx:
        return False
    expected = sign_transaction(tx, secret)
    return hmac.compare_digest(expected, tx["signature"])


def generate_transaction_hash(tx: Dict) -> str:
    payload = {
        k: v
        for k, v in tx.items()
        if k not in ["transaction_id", "signature", "validated"]
    }
    return generate_hash(payload)


# =========================
# TRANSACTION LIFECYCLE
# =========================

def build_transaction(
    client_id: str,
    round_number: int,
    model_hash: str,
    accuracy: float,
    f1_macro: float,
    roc_auc_ovr: float,
    ledger_path: str = LEDGER_PATH,
) -> Dict:
    ledger = load_ledger(ledger_path)
    last_tx = ledger[-1] if ledger else None

    previous_hash = last_tx["transaction_id"] if last_tx else "0" * 64
    references = [previous_hash] if last_tx else []

    tx = {
        "transaction_id": "",
        "client_id": client_id,
        "round_number": int(round_number),
        "model_hash": model_hash,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "roc_auc_ovr": float(roc_auc_ovr),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "previous_hash": previous_hash,
        "references": references,
        "validated": False,
        "signature": "",
    }

    tx["transaction_id"] = generate_transaction_hash(tx)
    secret = CLIENT_SECRETS.get(client_id, "")
    tx["signature"] = sign_transaction(tx, secret)

    return tx


def validate_transaction(tx: Dict, ledger: List[Dict]) -> Tuple[bool, str]:
    if any(existing.get("transaction_id") == tx.get("transaction_id") for existing in ledger):
        return False, "duplicate"

    secret = CLIENT_SECRETS.get(tx.get("client_id", ""), "")
    if not verify_signature(tx, secret):
        return False, "bad_signature"

    if float(tx.get("roc_auc_ovr", 0.0)) < CONSENSUS_ROC_AUC_MIN:
        return False, "consensus_reject"

    return True, "accepted"


def add_transaction(tx: Dict, ledger_path: str = LEDGER_PATH) -> Dict:
    ledger = load_ledger(ledger_path)
    is_valid, reason = validate_transaction(tx, ledger)
    tx["validated"] = bool(is_valid)

    ledger.append(tx)
    save_ledger(ledger, ledger_path)

    return {
        "validated": tx["validated"],
        "reason": reason,
    }
