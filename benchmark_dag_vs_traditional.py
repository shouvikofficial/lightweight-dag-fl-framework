"""
================================================================================
DAG VS. TRADITIONAL BLOCKCHAIN COMPREHENSIVE BENCHMARK (RIGOROUS MULTI-TRIAL)
================================================================================
Empirically benchmarks:
  1. Per-Transaction Confirmation Latency (Mean, Median, P95) across client scales (4 to 100 nodes)
  2. Transaction Throughput (Transactions Per Second / TPS)
  3. Cumulative 20-Round FL Commit Time (seconds)
  4. Serialized Ledger Storage Footprint on Disk (KB)

Methodology:
  - Multi-trial execution (N=5 runs per scale) reporting Mean ± Standard Deviation
  - Identical transaction payload distribution fed to both systems
  - Scientifically defensible controlled baseline modeling

Outputs:
  - Formatted publication table (Mean ± Std)
  - 4-Panel 300 DPI high-resolution figure:
    models/plots/figure_dag_vs_traditional_comprehensive.png

Usage:
  python benchmark_dag_vs_traditional.py
================================================================================
"""

import os
import sys
import time
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from blockchain.dag.dag_structure import DAG
from blockchain.transaction import Transaction
from blockchain.dag.dag_validator import DAGValidator

OUTPUT_DIR = "models/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. CONTROLLED TRADITIONAL LINEAR BLOCKCHAIN SIMULATOR
# ==============================================================================
class TraditionalBlock:
    """Represents a sequential block containing packaged transactions."""
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        data = f"{self.index}{self.previous_hash}{json.dumps(self.transactions, sort_keys=True)}{self.timestamp}{self.nonce}"
        return hashlib.sha256(data.encode()).hexdigest()

    def mine_block(self, difficulty=1):
        """Simulates lightweight PoW / PoS block validation overhead."""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()


class TraditionalBlockchain:
    """Controlled linear blockchain baseline with batch confirmation delay."""
    def __init__(self, max_tx_per_block=2, block_delay_ms=25):
        self.chain = [TraditionalBlock(0, "0", [{"msg": "Genesis"}])]
        self.pending_transactions = []
        self.max_tx_per_block = max_tx_per_block
        self.block_delay_sec = block_delay_ms / 1000.0

    def add_transaction(self, tx_data):
        self.pending_transactions.append(tx_data)
        if len(self.pending_transactions) >= self.max_tx_per_block:
            self._commit_block()

    def _commit_block(self):
        time.sleep(self.block_delay_sec)  # Network consensus & propagation latency
        new_block = TraditionalBlock(
            index=len(self.chain),
            previous_hash=self.chain[-1].hash,
            transactions=list(self.pending_transactions),
        )
        new_block.mine_block(difficulty=1)
        self.chain.append(new_block)
        self.pending_transactions = []

    def flush(self):
        if self.pending_transactions:
            self._commit_block()

    def get_storage_kb(self):
        serialized = json.dumps([{
            "index": b.index,
            "hash": b.hash,
            "prev_hash": b.previous_hash,
            "tx_count": len(b.transactions),
            "timestamp": b.timestamp,
        } for b in self.chain])
        return len(serialized.encode("utf-8")) / 1024.0


# ==============================================================================
# 2. HELPER: GENERATE IDENTICAL TRANSACTION PAYLOADS
# ==============================================================================
def generate_standard_transactions(num_tx, round_no=1):
    """Generates standardized FL model update transactions.
    round_no makes each round's transactions unique, preventing duplicate
    transaction IDs in the DAG ledger across FL rounds.
    """
    tx_list = []
    for i in range(num_tx):
        # Include round_no so each round produces unique model hashes
        weights_hash = hashlib.sha256(
            f"densenet201_round_{round_no}_client_{i}".encode()
        ).hexdigest()
        tx_dict = {
            "client_id": f"client_{i+1}",
            "model_hash": weights_hash,
            "accuracy": round(0.85 + 0.005 * (i % 10), 4),
            "round": round_no,
            "timestamp": time.time(),
        }
        tx_list.append(tx_dict)
    return tx_list


# ==============================================================================
# 3. BENCHMARK RUNNER (MULTI-TRIAL WITH P95 & MEAN ± STD)
# ==============================================================================
def run_rigorous_benchmark(trials=5):
    print("=" * 85)
    print(" 🚀 RIGOROUS BENCHMARK: PROPOSED DAG LEDGER VS. TRADITIONAL BLOCKCHAIN")
    print(f"    (Number of independent trials per configuration: N = {trials})")
    print("=" * 85)

    client_scales = [4, 8, 16, 32, 64, 100]

    # Metrics storage across scales
    results = {
        "scales": client_scales,
        "dag_lat_mean": [], "dag_lat_std": [],
        "dag_lat_p95": [],
        "trad_lat_mean": [], "trad_lat_std": [],
        "trad_lat_p95": [],
        "dag_tps_mean": [], "dag_tps_std": [],
        "trad_tps_mean": [], "trad_tps_std": [],
        "dag_storage_kb": [],
        "trad_storage_kb": [],
    }

    print(f"\n{'Clients':<8} | {'DAG Lat (ms)':<18} | {'Trad Lat (ms)':<18} | {'DAG TPS':<18} | {'Trad TPS':<18} | {'Lat. Speedup':<14}")
    print("-" * 105)

    for scale in client_scales:
        dag_lat_trials = []
        trad_lat_trials = []
        dag_tps_trials = []
        trad_tps_trials = []
        dag_p95_trials = []
        trad_p95_trials = []
        dag_stor_trials = []
        trad_stor_trials = []

        for _ in range(trials):
            raw_txs = generate_standard_transactions(scale)

            # ── 1. Benchmark Proposed DAG Ledger ─────────────────────────────
            dag = DAG()
            validator = DAGValidator(dag)
            tx_latencies_dag = []

            start_dag_total = time.perf_counter()
            for tx_data in raw_txs:
                t0 = time.perf_counter()
                tx_obj = Transaction(
                    client_id=tx_data["client_id"],
                    model_hash=tx_data["model_hash"],
                    accuracy=tx_data["accuracy"],
                )
                if validator.validate_transaction(tx_obj):
                    dag.add_transaction(tx_obj)
                    dag.validate_transaction(tx_obj.transaction_id)
                t1 = time.perf_counter()
                tx_latencies_dag.append((t1 - t0) * 1000.0)
            end_dag_total = time.perf_counter()

            dag_total_sec = end_dag_total - start_dag_total
            dag_lat_trials.append(np.mean(tx_latencies_dag))
            dag_p95_trials.append(np.percentile(tx_latencies_dag, 95))
            dag_tps_trials.append(scale / max(dag_total_sec, 1e-6))
            
            dag_dict_list = [node.to_dict() for node in dag.get_all_transactions()]
            dag_json = json.dumps(dag_dict_list)
            dag_stor_trials.append(len(dag_json.encode("utf-8")) / 1024.0)

            # ── 2. Benchmark Traditional Blockchain ──────────────────────────
            trad_chain = TraditionalBlockchain(max_tx_per_block=2, block_delay_ms=25)
            tx_latencies_trad = []

            start_trad_total = time.perf_counter()
            for tx_data in raw_txs:
                t0 = time.perf_counter()
                trad_chain.add_transaction(tx_data)
                t1 = time.perf_counter()
                tx_latencies_trad.append((t1 - t0) * 1000.0)
            trad_chain.flush()
            end_trad_total = time.perf_counter()

            trad_total_sec = end_trad_total - start_trad_total
            trad_lat_trials.append(np.mean(tx_latencies_trad))
            trad_p95_trials.append(np.percentile(tx_latencies_trad, 95))
            trad_tps_trials.append(scale / max(trad_total_sec, 1e-6))
            trad_stor_trials.append(trad_chain.get_storage_kb())

        d_lat_m, d_lat_s = np.mean(dag_lat_trials), np.std(dag_lat_trials)
        t_lat_m, t_lat_s = np.mean(trad_lat_trials), np.std(trad_lat_trials)
        d_tps_m, d_tps_s = np.mean(dag_tps_trials), np.std(dag_tps_trials)
        t_tps_m, t_tps_s = np.mean(trad_tps_trials), np.std(trad_tps_trials)
        speedup = t_lat_m / max(d_lat_m, 1e-6)

        results["dag_lat_mean"].append(d_lat_m)
        results["dag_lat_std"].append(d_lat_s)
        results["dag_lat_p95"].append(np.mean(dag_p95_trials))

        results["trad_lat_mean"].append(t_lat_m)
        results["trad_lat_std"].append(t_lat_s)
        results["trad_lat_p95"].append(np.mean(trad_p95_trials))

        results["dag_tps_mean"].append(d_tps_m)
        results["dag_tps_std"].append(d_tps_s)
        results["trad_tps_mean"].append(t_tps_m)
        results["trad_tps_std"].append(t_tps_s)

        results["dag_storage_kb"].append(np.mean(dag_stor_trials))
        results["trad_storage_kb"].append(np.mean(trad_stor_trials))

        lat_speedup_label = f"{speedup:.1f}x"
        print(f"{scale:<8} | {d_lat_m:6.3f} ± {d_lat_s:5.3f} ms | {t_lat_m:6.3f} ± {t_lat_s:5.3f} ms | {d_tps_m:7.1f} ± {d_tps_s:5.1f} | {t_tps_m:6.1f} ± {t_tps_s:4.1f} | {lat_speedup_label:<14}")

    print("-" * 105)

    # ==============================================================================
    # 4. BENCHMARK CUMULATIVE 20-ROUND FL COMMIT TIME
    # ==============================================================================
    print("\n⏱️  Benchmarking Cumulative 20-Round FL Commit Time (4 Clients per round)...")
    fl_rounds = 20
    clients_per_round = 4

    dag_round_commit_cum = []
    trad_round_commit_cum = []

    dag_fl = DAG()
    val_fl = DAGValidator(dag_fl)
    trad_fl = TraditionalBlockchain(max_tx_per_block=2, block_delay_ms=25)

    cum_dag = 0.0
    cum_trad = 0.0

    for r in range(1, fl_rounds + 1):
        round_txs = generate_standard_transactions(clients_per_round, round_no=r)

        t0 = time.perf_counter()
        for tx_data in round_txs:
            tx_obj = Transaction(client_id=tx_data["client_id"], model_hash=tx_data["model_hash"], accuracy=tx_data["accuracy"])
            if val_fl.validate_transaction(tx_obj):
                dag_fl.add_transaction(tx_obj)
                dag_fl.validate_transaction(tx_obj.transaction_id)
        t1 = time.perf_counter()
        cum_dag += (t1 - t0)
        dag_round_commit_cum.append(cum_dag)

        t0 = time.perf_counter()
        for tx_data in round_txs:
            trad_fl.add_transaction(tx_data)
        trad_fl.flush()
        t1 = time.perf_counter()
        cum_trad += (t1 - t0)
        trad_round_commit_cum.append(cum_trad)

    # ==============================================================================
    # 5. GENERATE 4-PANEL PUBLICATION FIGURE
    # ==============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (a) Latency Comparison
    ax1.plot(client_scales, results["dag_lat_mean"], color='#28a745', marker='o', linewidth=2.2, label='Proposed DAG Ledger (Mean)')
    ax1.plot(client_scales, results["dag_lat_p95"], color='#20c997', linestyle=':', marker='.', linewidth=1.5, label='Proposed DAG (P95 Latency)')
    ax1.plot(client_scales, results["trad_lat_mean"], color='#dc3545', marker='s', linestyle='--', linewidth=2.2, label='Traditional Blockchain Simulator (Mean)')
    ax1.set_title('(a) Transaction Processing Latency vs. Node Scale', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax1.set_ylabel('Per-Transaction Processing Latency (ms)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.5)

    # (b) Throughput Comparison (TPS)
    ax2.plot(client_scales, results["dag_tps_mean"], color='#007bff', marker='^', linewidth=2.2, label='Proposed DAG Ledger')
    ax2.plot(client_scales, results["trad_tps_mean"], color='#dc3545', marker='x', linestyle='--', linewidth=2.2, label='Traditional Blockchain Simulator')
    ax2.set_title('(b) Transaction Throughput (TPS) Scalability', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax2.set_ylabel('Throughput (Transactions / Sec)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8.5)

    # (c) Cumulative 20-Round FL Commit Time
    rounds_axis = list(range(1, fl_rounds + 1))
    ax3.plot(rounds_axis, dag_round_commit_cum, color='#28a745', marker='o', linewidth=2.0, label='Proposed DAG Ledger')
    ax3.plot(rounds_axis, trad_round_commit_cum, color='#dc3545', marker='s', linestyle='--', linewidth=2.0, label='Traditional Blockchain Simulator')
    ax3.set_title('(c) Cumulative Ledger Commit Time across 20 FL Rounds', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Federated Learning Round', fontsize=10)
    ax3.set_ylabel('Cumulative Commit Overhead (Seconds)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper left', fontsize=8.5)

    # (d) Ledger Storage Footprint (KB)
    width = 0.35
    x_indices = np.arange(len(client_scales))
    ax4.bar(x_indices - width/2, results["dag_storage_kb"], width, label='Proposed DAG Ledger', color='#17a2b8')
    ax4.bar(x_indices + width/2, results["trad_storage_kb"], width, label='Traditional Blockchain Simulator', color='#6c757d')
    ax4.set_title('(d) Serialized Ledger Storage Footprint vs. Node Scale', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax4.set_ylabel('Serialized Ledger Size (KB)', fontsize=10)
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(client_scales)
    ax4.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax4.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "figure_dag_vs_traditional_comprehensive.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 4-Panel High-Resolution Publication Plot saved successfully:")
    print(f"   -> {plot_path}")
    print("=" * 85)


# ==============================================================================
# 6. REAL ETHEREUM BASELINE  (Ganache local node via web3.py)
# ==============================================================================
GANACHE_RPC = "http://127.0.0.1:8545"

def _ganache_available():
    """Return True only if a Ganache node is reachable."""
    try:
        # pyrefly: ignore [missing-import]
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(GANACHE_RPC, request_kwargs={"timeout": 3}))
        _ = w3.eth.accounts          # raises if not connected
        return True
    except Exception:
        return False


class RealEthereumBlockchain:
    """
    Real Ethereum blockchain baseline using a local Ganache node (web3.py).
    Ganache accounts are pre-unlocked, so no manual private-key signing needed.
    """
    def __init__(self):
        # pyrefly: ignore [missing-import]
        from web3 import Web3
        self.w3 = Web3(Web3.HTTPProvider(GANACHE_RPC))
        self.sender   = self.w3.eth.accounts[0]
        self.receiver = self.w3.eth.accounts[1]

    def add_transaction(self, tx_data):
        """Submit one real on-chain transaction encoding the FL model hash as calldata.
        
        Latency measured = end-to-end local confirmation latency:
          Web3 RPC call → Ganache EVM execution → block inclusion → receipt.
        NOTE: This includes more processing than DAG (gas validation, EVM overhead).
        Measurements reflect local Ganache conditions, NOT Ethereum mainnet performance.
        """
        data_hex = "0x" + tx_data["model_hash"]          # 64 hex chars = 32 bytes
        tx_hash  = self.w3.eth.send_transaction({
            "from":  self.sender,
            "to":    self.receiver,
            "value": 0,
            "gas":   50_000,
            "data":  data_hex,
        })
        # blockTime 0 → instant mining; wait for receipt to confirm latency
        self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=10)

    def get_storage_kb(self):
        """Measure actual on-chain storage by reading every block from the node."""
        total = 0
        for i in range(self.w3.eth.block_number + 1):
            blk    = self.w3.eth.get_block(i, full_transactions=True)
            total += len(str(dict(blk)))
        return total / 1024.0


def run_ganache_supplement_benchmark(trials=3):
    """
    Supplementary benchmark: Proposed DAG Ledger vs. Real Ethereum (Ganache).
    Runs fewer trials than the main benchmark to keep execution time reasonable.
    """
    if not _ganache_available():
        print("\n⚠️  Ganache not reachable at http://127.0.0.1:8545 — skipping real Ethereum benchmark.")
        print("    Start Ganache with:  ganache --port 8545 --blockTime 0 --deterministic")
        return

    print("\n" + "=" * 85)
    print(" ⛓️  SUPPLEMENTARY BENCHMARK: PROPOSED DAG LEDGER  VS.  LOCAL ETHEREUM-COMPATIBLE BASELINE (Ganache)")
    print(f"    RPC endpoint : {GANACHE_RPC}")
    print(f"    Trials       : N = {trials}  |  blockTime = 0  |  Environment: local machine only")
    print(f"    NOTE: Measures end-to-end local confirmation latency — not public Ethereum mainnet performance.")
    print("=" * 85)

    client_scales = [4, 8, 16, 32, 64, 100]

    g_lat_mean, g_lat_std = [], []
    g_tps_mean, g_tps_std = [], []
    g_storage_kb          = []
    d_lat_mean_g, d_tps_mean_g, d_storage_kb_g = [], [], []

    print(f"\n{'Clients':<8} | {'DAG Lat (ms)':<18} | {'ETH Lat (ms)':<18} | {'DAG TPS':<15} | {'ETH TPS':<15} | {'Lat. Speedup':<14}")
    print("-" * 100)

    for scale in client_scales:
        dag_lats, eth_lats   = [], []
        dag_tps_t, eth_tps_t = [], []
        dag_stor_t, eth_stor_t = [], []

        for _ in range(trials):
            raw_txs = generate_standard_transactions(scale)

            # ── Proposed DAG Ledger ──────────────────────────────────────────
            dag = DAG();  val = DAGValidator(dag);  per_tx = []
            t_start = time.perf_counter()
            for tx_data in raw_txs:
                t0  = time.perf_counter()
                tx  = Transaction(client_id=tx_data["client_id"],
                                  model_hash=tx_data["model_hash"],
                                  accuracy=tx_data["accuracy"])
                if val.validate_transaction(tx):
                    dag.add_transaction(tx)
                    dag.validate_transaction(tx.transaction_id)
                t1  = time.perf_counter()
                per_tx.append((t1 - t0) * 1000.0)
            t_end = time.perf_counter()
            dag_lats.append(np.mean(per_tx))
            dag_tps_t.append(scale / max(t_end - t_start, 1e-6))
            dag_stor_t.append(len(json.dumps(
                [n.to_dict() for n in dag.get_all_transactions()]
            ).encode()) / 1024.0)

            # ── Real Ethereum (Ganache) ──────────────────────────────────────
            eth_chain = RealEthereumBlockchain()
            eth_per_tx = []
            t_start = time.perf_counter()
            for tx_data in raw_txs:
                t0 = time.perf_counter()
                eth_chain.add_transaction(tx_data)
                t1 = time.perf_counter()
                eth_per_tx.append((t1 - t0) * 1000.0)
            t_end = time.perf_counter()
            eth_lats.append(np.mean(eth_per_tx))
            eth_tps_t.append(scale / max(t_end - t_start, 1e-6))
            eth_stor_t.append(eth_chain.get_storage_kb())

        d_m, d_s  = np.mean(dag_lats),  np.std(dag_lats)
        e_m, e_s  = np.mean(eth_lats),  np.std(eth_lats)
        dt_m, dt_s = np.mean(dag_tps_t), np.std(dag_tps_t)
        et_m, et_s = np.mean(eth_tps_t), np.std(eth_tps_t)
        speedup    = e_m / max(d_m, 1e-6)

        d_lat_mean_g.append(d_m);   d_tps_mean_g.append(dt_m)
        d_storage_kb_g.append(np.mean(dag_stor_t))
        g_lat_mean.append(e_m);     g_lat_std.append(e_s)
        g_tps_mean.append(et_m);    g_tps_std.append(et_s)
        g_storage_kb.append(np.mean(eth_stor_t))

        print(f"{scale:<8} | {d_m:6.3f} ± {d_s:5.3f} ms | {e_m:6.1f} ± {e_s:5.1f} ms | "
              f"{dt_m:8.1f} ± {dt_s:5.1f} | {et_m:6.1f} ± {et_s:4.1f} | {speedup:.1f}x")

    print("-" * 100)

    # ── 20-Round Cumulative FL Commit Time (DAG vs. Ganache) ─────────────────
    print("\n⏱️  Benchmarking Cumulative 20-Round FL Commit Time (DAG vs. Ganache)...")
    fl_rounds = 20
    clients_per_round = 4

    dag_round_commit_cum = []
    eth_round_commit_cum = []

    dag_fl = DAG()
    val_fl = DAGValidator(dag_fl)
    eth_fl = RealEthereumBlockchain()

    cum_dag = 0.0
    cum_eth = 0.0

    for r in range(1, fl_rounds + 1):
        round_txs = generate_standard_transactions(clients_per_round, round_no=r)

        # DAG commit
        t0 = time.perf_counter()
        for tx_data in round_txs:
            tx_obj = Transaction(client_id=tx_data["client_id"], model_hash=tx_data["model_hash"], accuracy=tx_data["accuracy"])
            if val_fl.validate_transaction(tx_obj):
                dag_fl.add_transaction(tx_obj)
                dag_fl.validate_transaction(tx_obj.transaction_id)
        t1 = time.perf_counter()
        cum_dag += (t1 - t0)
        dag_round_commit_cum.append(cum_dag)

        # Ganache Ethereum commit
        t0 = time.perf_counter()
        for tx_data in round_txs:
            eth_fl.add_transaction(tx_data)
        t1 = time.perf_counter()
        cum_eth += (t1 - t0)
        eth_round_commit_cum.append(cum_eth)

    # ── Comprehensive 4-Panel Figure: DAG vs. Real Ethereum (Ganache) ────────
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # (a) Latency Comparison
    ax1.plot(client_scales, d_lat_mean_g, color='#28a745', marker='o', linewidth=2.2, label='Proposed DAG Ledger')
    ax1.plot(client_scales, g_lat_mean, color='#fd7e14', marker='D', linestyle='--', linewidth=2.2, label='Local Ethereum Baseline (Ganache)')
    ax1.fill_between(client_scales,
                     np.array(g_lat_mean) - np.array(g_lat_std),
                     np.array(g_lat_mean) + np.array(g_lat_std),
                     color='#fd7e14', alpha=0.15, label='Ganache ± 1 Std Dev')
    ax1.set_title('(a) Transaction Processing Latency vs. Node Scale', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax1.set_ylabel('Per-Transaction Processing Latency (ms)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.5)

    # (b) Throughput Comparison (TPS)
    ax2.plot(client_scales, d_tps_mean_g, color='#007bff', marker='^', linewidth=2.2, label='Proposed DAG Ledger')
    ax2.plot(client_scales, g_tps_mean, color='#fd7e14', marker='D', linestyle='--', linewidth=2.2, label='Local Ethereum Baseline (Ganache)')
    ax2.set_title('(b) Transaction Throughput (TPS) Scalability', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax2.set_ylabel('Throughput (Transactions / Sec)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8.5)
    ax2.text(0.01, 0.35,
             'DAG: in-memory implementation\nGanache: Web3 RPC + EVM block confirmation\n(Local environment only)',
             transform=ax2.transAxes, fontsize=7.5, color='#555555',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', alpha=0.7))

    # (c) Cumulative 20-Round FL Commit Time
    rounds_axis = list(range(1, fl_rounds + 1))
    ax3.plot(rounds_axis, dag_round_commit_cum, color='#28a745', marker='o', linewidth=2.0, label='Proposed DAG Ledger')
    ax3.plot(rounds_axis, eth_round_commit_cum, color='#fd7e14', marker='D', linestyle='--', linewidth=2.0, label='Local Ethereum Baseline (Ganache)')
    ax3.set_title('(c) Cumulative Ledger Commit Time across 20 FL Rounds', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Federated Learning Round', fontsize=10)
    ax3.set_ylabel('Cumulative Commit Overhead (Seconds)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper left', fontsize=8.5)

    # (d) Serialized Storage Footprint (KB)
    width = 0.35
    x_indices = np.arange(len(client_scales))
    ax4.bar(x_indices - width/2, d_storage_kb_g, width, label='Proposed DAG Ledger', color='#17a2b8')
    ax4.bar(x_indices + width/2, g_storage_kb, width, label='Local Ethereum Baseline (Ganache)', color='#fd7e14')
    ax4.set_title('(d) Serialized Storage Footprint vs. Node Scale', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Number of Federated Clients (Nodes)', fontsize=10)
    ax4.set_ylabel('Serialized Storage Size (KB)', fontsize=10)
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(client_scales)
    ax4.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax4.legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    supp_path = os.path.join(OUTPUT_DIR, "figure_dag_vs_real_ethereum_ganache.png")
    plt.savefig(supp_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 4-Panel High-Resolution Ganache Publication Plot saved:")
    print(f"   -> {supp_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_rigorous_benchmark(trials=5)
    run_ganache_supplement_benchmark(trials=3)
