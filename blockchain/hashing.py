import hashlib
import json


def generate_hash(data):
    """
    Generate SHA256 hash from dictionary/string data.
    """

    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)

    return hashlib.sha256(data.encode()).hexdigest()


def verify_hash(data, old_hash):
    """
    Verify whether hash matches original data.
    """

    new_hash = generate_hash(data)

    return new_hash == old_hash