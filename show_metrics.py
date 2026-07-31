"""
show_metrics.py
───────────────
Blockchain + FL Performance Metrics — Terminal Report
Reads blockchain/ledger.json and prints all 8 metrics to the console.

Usage:
    python show_metrics.py
    python show_metrics.py --ledger blockchain/ledger.json
"""

import argparse
import hashlib
import hmac
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── rich imports ───────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_ts(ts_str: str):
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(ts_str), fmt)
        except ValueError:
            continue
    return None


def load_ledger(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _section(title: str):
    if HAS_RICH:
        console.print(f"\n[bold cyan]{'─'*70}[/bold cyan]")
        console.print(f"[bold white]  {title}[/bold white]")
        console.print(f"[bold cyan]{'─'*70}[/bold cyan]")
    else:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")


def _kv(label: str, value: str, note: str = ""):
    if HAS_RICH:
        note_part = f"  [dim]{note}[/dim]" if note else ""
        console.print(f"  [bold yellow]{label:<38}[/bold yellow] [green]{value}[/green]{note_part}")
    else:
        print(f"  {label:<38} {value}  {note}")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(ledger: list) -> dict:
    results = {}
    dummy_secret = "benchmark_secret"

    # ── 1. Latency ─────────────────────────────────────────────────────────────
    latencies_ms = []
    for tx in ledger:
        payload = json.dumps(
            {k: v for k, v in tx.items() if k != "signature"},
            sort_keys=True, separators=(",", ":")
        ).encode()
        t0 = time.perf_counter()
        hashlib.sha256(payload).hexdigest()
        json.dumps(tx)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000)

    results["latency"] = {
        "avg_ms":  float(np.mean(latencies_ms)),
        "min_ms":  float(np.min(latencies_ms)),
        "max_ms":  float(np.max(latencies_ms)),
        "p95_ms":  float(np.percentile(latencies_ms, 95)),
        "p99_ms":  float(np.percentile(latencies_ms, 99)),
        "samples": len(latencies_ms),
    }

    # ── 2. Throughput (TPS) ────────────────────────────────────────────────────
    # Reuse latencies_ms already measured in section 1 — no second benchmark loop.
    # This keeps TPS directly consistent with the reported latency value:
    #   TPS = 1000 / avg_latency_ms
    # (Using calendar timestamps from the ledger would give nonsensical results
    #  like 0.00003 TPS because training sessions ran on different days.)

    # Group transactions by round; track index into latencies_ms for per-round TPS
    round_buckets: dict = {}      # rn -> list of tx dicts (reused for consensus)
    round_latency_idx: dict = {}  # rn -> list of indices into latencies_ms
    for i, tx in enumerate(ledger):
        rn = tx.get("round_number")
        if rn is not None:
            round_buckets.setdefault(rn, []).append(tx)
            round_latency_idx.setdefault(rn, []).append(i)

    # Overall TPS: derived directly from the latency already measured above
    avg_bench_ms = results["latency"]["avg_ms"]
    overall_tps  = 1_000.0 / avg_bench_ms if avg_bench_ms > 0 else 0.0

    # Per-round TPS: sum per-tx latencies within that round from latencies_ms
    per_round_tps = {}
    for rn, indices in sorted(round_latency_idx.items()):
        round_total_ms = sum(latencies_ms[i] for i in indices)
        per_round_tps[rn] = (len(indices) / round_total_ms) * 1_000 if round_total_ms > 0 else float(len(indices))

    results["throughput"] = {
        "overall_tps":   overall_tps,
        "per_round_tps": per_round_tps,
        "total_tx":      len(ledger),
        "avg_bench_ms":  avg_bench_ms,
        "method":        "1000 / avg_latency_ms  (reuses section-1 measurements, no double benchmark)",
    }

    # ── 3. Communication Overhead ──────────────────────────────────────────────
    tx_sizes = [sys.getsizeof(json.dumps(tx)) for tx in ledger]
    results["communication"] = {
        "total_kb":        sum(tx_sizes) / 1024,
        "avg_bytes":       float(np.mean(tx_sizes)),
        "max_bytes":       float(np.max(tx_sizes)),
        "hash_overhead_b": 64 * 2,
    }

    # ── 4. Storage Overhead ────────────────────────────────────────────────────
    ledger_file_size = Path("blockchain/ledger.json").stat().st_size if Path("blockchain/ledger.json").exists() else 0
    hash_bytes = sum(
        len(tx.get("transaction_id", "")) +
        len(tx.get("model_hash", "")) +
        len(tx.get("previous_hash", "")) +
        len(tx.get("signature", "")) +
        sum(len(r) for r in tx.get("references", []))
        for tx in ledger
    )
    results["storage"] = {
        "ledger_file_kb": ledger_file_size / 1024,
        "hash_data_kb":   hash_bytes / 1024,
        "metadata_kb":    (sum(tx_sizes) - hash_bytes) / 1024,
        "tx_count":       len(ledger),
    }

    # ── 5. Verification Time ───────────────────────────────────────────────────
    verify_ms = []
    for tx in ledger:
        payload = json.dumps(
            {k: v for k, v in tx.items() if k != "signature"},
            sort_keys=True, separators=(",", ":")
        ).encode()
        t0 = time.perf_counter()
        hmac.new(dummy_secret.encode(), payload, hashlib.sha256).hexdigest()
        t1 = time.perf_counter()
        verify_ms.append((t1 - t0) * 1_000)

    results["verification"] = {
        "avg_ms": float(np.mean(verify_ms)),
        "p99_ms": float(np.percentile(verify_ms, 99)),
        "total":  len(verify_ms),
    }

    # ── 6. Consensus Time ──────────────────────────────────────────────────────
    # Consensus = time to verify + hash ALL clients' transactions in one round.
    # We measure this via perf_counter, NOT calendar timestamps.
    # Calendar timestamps span weeks (training sessions on different days),
    # which would give absurd results like "consensus took 66 days".
    consensus = {}
    for rn, txs in sorted(round_buckets.items()):
        # Simulate the server receiving and validating every client tx in the round
        t_start = time.perf_counter()
        for tx in txs:
            payload = json.dumps(
                {k: v for k, v in tx.items() if k != "signature"},
                sort_keys=True, separators=(",", ":")
            ).encode()
            hashlib.sha256(payload).hexdigest()                          # hash
            hmac.new(dummy_secret.encode(), payload, hashlib.sha256).hexdigest()  # verify
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1_000
        consensus[rn] = {
            "clients":      len(txs),
            "consensus_ms": round(elapsed_ms, 4),
            "consensus_s":  round(elapsed_ms / 1_000, 6),
        }

    consensus_ms_vals = [v["consensus_ms"] for v in consensus.values()]
    results["consensus"] = {
        "per_round": consensus,
        "avg_ms":    float(np.mean(consensus_ms_vals)) if consensus_ms_vals else 0.0,
        "max_ms":    float(np.max(consensus_ms_vals))  if consensus_ms_vals else 0.0,
        "avg_s":     float(np.mean(consensus_ms_vals)) / 1_000 if consensus_ms_vals else 0.0,
        "max_s":     float(np.max(consensus_ms_vals))  / 1_000 if consensus_ms_vals else 0.0,
        "method":    "Benchmark (perf_counter over hash+verify per round) — not calendar timestamps",
    }

    # ── 7. Scalability ─────────────────────────────────────────────────────────
    avg_tx_ms = results["latency"]["avg_ms"] + results["verification"]["avg_ms"]
    scale_rows = []
    for n in [1, 2, 4, 5, 10, 20, 50]:
        t_s  = (avg_tx_ms * n) / 1_000
        tps  = n / t_s if t_s > 0 else float(n)
        stor = (results["communication"]["avg_bytes"] * n) / 1024
        scale_rows.append({
            "clients":      n,
            "round_time_s": round(t_s, 5),
            "tps":          round(tps, 2),
            "storage_kb":   round(stor, 2),
        })
    results["scalability"] = scale_rows

    # ── 8. Computation Cost ────────────────────────────────────────────────────
    ops = {}
    sample = json.dumps(ledger[0], sort_keys=True).encode()

    t0 = time.perf_counter()
    for _ in range(1_000):
        hashlib.sha256(sample).hexdigest()
    ops["SHA-256 Hash"] = {"iterations": 1000, "total_ms": (time.perf_counter() - t0) * 1000}

    t0 = time.perf_counter()
    for _ in range(1_000):
        hmac.new(dummy_secret.encode(), sample, hashlib.sha256).hexdigest()
    ops["HMAC-SHA256 Verify"] = {"iterations": 1000, "total_ms": (time.perf_counter() - t0) * 1000}

    t0 = time.perf_counter()
    for _ in range(1_000):
        json.dumps(ledger[0])
    ops["JSON Serialize (1 tx)"] = {"iterations": 1000, "total_ms": (time.perf_counter() - t0) * 1000}

    t0 = time.perf_counter()
    for _ in range(10):
        json.dumps(ledger)
    ops["JSON Serialize (full ledger)"] = {"iterations": 10, "total_ms": (time.perf_counter() - t0) * 1000}

    for v in ops.values():
        v["per_op_ms"]   = v["total_ms"] / v["iterations"]
        v["ops_per_sec"] = int(1_000 / v["per_op_ms"]) if v["per_op_ms"] > 0 else 0

    results["computation"] = ops
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_metrics(m: dict, ledger_path: str, ledger: list):
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold white]  Blockchain + FL — Performance Metrics Report  [/bold white]\n"
            f"  [dim]Ledger : {ledger_path}   |   Transactions : {len(ledger)}[/dim]",
            border_style="bright_blue", padding=(0, 2),
        ))
    else:
        print("\n" + "═"*70)
        print("  Blockchain + FL — Performance Metrics Report")
        print(f"  Ledger: {ledger_path}   |   Transactions: {len(ledger)}")
        print("═"*70)

    # 1 · Latency
    _section("⏱   1 · LATENCY  (verify + record one model update)")
    lat = m["latency"]
    _kv("Avg Latency",            f"{lat['avg_ms']:.4f} ms")
    _kv("Min Latency",            f"{lat['min_ms']:.4f} ms")
    _kv("Max Latency",            f"{lat['max_ms']:.4f} ms")
    _kv("P95 Latency",            f"{lat['p95_ms']:.4f} ms")
    _kv("P99 Latency",            f"{lat['p99_ms']:.4f} ms")
    _kv("Transactions measured",  str(lat["samples"]))

    # 2 · Throughput
    _section("⚡   2 · THROUGHPUT  (model updates per second — benchmark-based)")
    thr = m["throughput"]
    _kv("Overall TPS",               f"{thr['overall_tps']:,.2f} tx/s")
    _kv("Avg processing time / tx",  f"{thr['avg_bench_ms']:.4f} ms")
    _kv("Total transactions",         str(thr["total_tx"]))
    _kv("Method",                     thr["method"])
    if HAS_RICH:
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", padding=(0, 2))
        t.add_column("FL Round", style="cyan",  justify="right")
        t.add_column("Txs in Round", style="yellow", justify="right")
        t.add_column("TPS (benchmark)", style="green", justify="right")
        for rn, tps in sorted(thr["per_round_tps"].items()):
            n_txs = len([tx for tx in ledger if tx.get("round_number") == rn])
            t.add_row(str(rn), str(n_txs), f"{tps:,.2f}")
        console.print(t)
    else:
        for rn, tps in sorted(thr["per_round_tps"].items()):
            print(f"    Round {rn}: {tps:,.2f} tx/s")

    # 3 · Communication
    _section("📡   3 · COMMUNICATION OVERHEAD  (data transmitted per update)")
    com = m["communication"]
    _kv("Total transmitted (all txs)", f"{com['total_kb']:.2f} KB")
    _kv("Avg per transaction",          f"{com['avg_bytes']:.0f} bytes")
    _kv("Max single transaction",       f"{com['max_bytes']:.0f} bytes")
    _kv("Hash overhead per tx",         f"{com['hash_overhead_b']} bytes  (tx_id + signature)")

    # 4 · Storage
    _section("💾   4 · STORAGE OVERHEAD  (hashes + transactions + blocks)")
    sto = m["storage"]
    _kv("Ledger file size",    f"{sto['ledger_file_kb']:.1f} KB")
    _kv("Hash data total",     f"{sto['hash_data_kb']:.1f} KB")
    _kv("Metadata total",      f"{sto['metadata_kb']:.1f} KB")
    _kv("Transaction count",   str(sto["tx_count"]))

    # 5 · Verification
    _section("🔐   5 · VERIFICATION TIME  (HMAC-SHA256 per transaction)")
    ver = m["verification"]
    _kv("Avg verification time", f"{ver['avg_ms']:.4f} ms")
    _kv("P99 verification time", f"{ver['p99_ms']:.4f} ms")
    _kv("Total verified",         str(ver["total"]))

    # 6 · Consensus
    _section("🤝   6 · CONSENSUS TIME  (hash+verify all clients per round — benchmark)")
    con = m["consensus"]
    _kv("Avg consensus time", f"{con['avg_ms']:.4f} ms")
    _kv("Max consensus time", f"{con['max_ms']:.4f} ms")
    _kv("Method",              con["method"])
    if HAS_RICH:
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", padding=(0, 2))
        t.add_column("Round",            style="cyan",   justify="right")
        t.add_column("Clients",          style="yellow", justify="right")
        t.add_column("Consensus (ms)",   style="green",  justify="right")
        for rn, d in sorted(con["per_round"].items()):
            t.add_row(str(rn), str(d["clients"]), f"{d['consensus_ms']:.4f}")
        console.print(t)
    else:
        for rn, d in sorted(con["per_round"].items()):
            print(f"    Round {rn}: {d['clients']} clients  →  {d['consensus_ms']:.4f} ms")

    # 7 · Scalability
    _section("📊   7 · SCALABILITY  (performance estimate vs client count)")
    if HAS_RICH:
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta", padding=(0, 2))
        t.add_column("Clients",             style="cyan",   justify="right")
        t.add_column("Est. Round Time (s)", style="yellow", justify="right")
        t.add_column("Est. TPS",            style="green",  justify="right")
        t.add_column("Est. Storage/Round",  style="blue",   justify="right")
        for row in m["scalability"]:
            marker = " ◀ current" if row["clients"] == 4 else ""
            t.add_row(
                str(row["clients"]) + marker,
                f"{row['round_time_s']:.5f}",
                f"{row['tps']:.2f}",
                f"{row['storage_kb']:.2f} KB",
            )
        console.print(t)
    else:
        print(f"  {'Clients':<12} {'Round Time(s)':<18} {'TPS':<12} Storage/Round")
        for row in m["scalability"]:
            marker = " ◀" if row["clients"] == 4 else ""
            print(f"  {str(row['clients'])+marker:<12} {row['round_time_s']:<18.5f} {row['tps']:<12.2f} {row['storage_kb']:.2f} KB")

    # 8 · Computation Cost
    _section("💻   8 · COMPUTATION COST  (CPU time per blockchain operation)")
    if HAS_RICH:
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta", padding=(0, 2))
        t.add_column("Operation",   style="cyan")
        t.add_column("Per-op (ms)", style="green",  justify="right")
        t.add_column("Ops / sec",   style="yellow", justify="right")
        t.add_column("Batch total", style="blue",   justify="right")
        for op, v in m["computation"].items():
            t.add_row(op,
                      f"{v['per_op_ms']:.5f}",
                      f"{v['ops_per_sec']:,}",
                      f"{v['total_ms']:.2f} ms  (×{v['iterations']})")
        console.print(t)
    else:
        print(f"  {'Operation':<35} {'Per-op (ms)':<16} {'Ops/sec':<12} Batch")
        for op, v in m["computation"].items():
            print(f"  {op:<35} {v['per_op_ms']:<16.5f} {v['ops_per_sec']:<12,} {v['total_ms']:.2f} ms (×{v['iterations']})")

    # 9 · Trust & Security Evaluation
    _section("🛡️   9 · 4-FACTOR ADAPTIVE TRUST & 3-TIER SECURITY EVALUATION")
    trust_file = Path("logs/trust_scores.json")
    if trust_file.exists():
        try:
            with open(trust_file, "r", encoding="utf-8") as f:
                t_log = json.load(f)
            if t_log:
                latest = t_log[-1].get("client_trust", {})
                _kv("FL Rounds logged", str(len(t_log)))
                _kv("Active clients evaluated", str(len(latest)))
                _kv("Trust Formula", "0.35*Hist + 0.25*Sim + 0.20*Acc + 0.20*BC")
                _kv("3-Tier System", "T >= 0.80: ACCEPT | 0.50-0.79: PENALIZE | T < 0.50: REJECT")
                if HAS_RICH:
                    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", padding=(0, 2))
                    t.add_column("Client ID", style="cyan")
                    t.add_column("Similarity", style="yellow", justify="right")
                    t.add_column("Val Acc", style="blue", justify="right")
                    t.add_column("Trust Score", style="green", justify="right")
                    t.add_column("3-Tier Action", style="bold white", justify="center")
                    for cid, d in sorted(latest.items()):
                        act = d.get("action", "ACCEPT" if d.get("accepted") else "REJECT")
                        if act == "ACCEPT":
                            st_color = "[bold green]ACCEPT (Full)[/bold green]"
                        elif act == "PENALIZE":
                            st_color = "[bold yellow]PENALIZE (50%)[/bold yellow]"
                        else:
                            st_color = "[bold red]REJECT (0%)[/bold red]"
                        t.add_row(cid, f"{d.get('similarity', 0.0):.4f}", f"{d.get('val_accuracy', 0.8):.4f}", f"{d['trust_score']:.4f}", st_color)
                    console.print(t)
                else:
                    for cid, d in sorted(latest.items()):
                        act = d.get("action", "ACCEPT" if d.get("accepted") else "REJECT")
                        print(f"    {cid:<12} Similarity: {d.get('similarity',0.0):.4f} | Acc: {d.get('val_accuracy',0.8):.4f} | Trust: {d['trust_score']:.4f} | Action: {act}")
            else:
                _kv("Trust Status", "No trust log entries yet (run FL server)")
        except Exception as e:
            _kv("Trust Log Error", str(e))
    else:
        _kv("Trust Status", "logs/trust_scores.json not found (will be generated upon running FL server)")

    # Summary
    _section("📋   SUMMARY — All Metrics at a Glance")
    cpu = m["computation"]["SHA-256 Hash"]
    rows = [
        ("Latency (avg)",                    f"{lat['avg_ms']:.4f} ms",              "Time to verify + record one update"),
        ("Latency (P95)",                    f"{lat['p95_ms']:.4f} ms",              "95th-percentile tail latency"),
        ("Throughput (TPS)",                 f"{thr['overall_tps']:,.2f} tx/s",      "Benchmark-based (not calendar time)"),
        ("Communication overhead (avg/tx)",  f"{com['avg_bytes']:.0f} bytes",        "Bytes sent per blockchain tx"),
        ("Storage overhead (ledger total)",  f"{sto['ledger_file_kb']:.1f} KB",      "Disk used by full ledger"),
        ("Verification time (avg)",          f"{ver['avg_ms']:.4f} ms",              "HMAC-SHA256 per tx"),
        ("Consensus time (avg)",             f"{con['avg_ms']:.4f} ms",              "Benchmark: hash+verify per round"),
        ("Computation cost (SHA-256/op)",    f"{cpu['per_op_ms']:.5f} ms",           "CPU cost per hash operation"),
    ]

    if HAS_RICH:
        t = Table(box=box.ROUNDED, show_header=True,
                  header_style="bold white on dark_blue",
                  padding=(0, 2), border_style="bright_blue")
        t.add_column("Metric",           style="bold cyan",  min_width=36)
        t.add_column("Value",            style="bold green", justify="right", min_width=18)
        t.add_column("What it Measures", style="dim white",  min_width=36)
        for r in rows:
            t.add_row(*r)
        console.print(t)
        console.print(Panel(
            f"[bold green]✔  Complete.[/bold green]  "
            f"[dim]{len(ledger)} transactions analysed.[/dim]",
            border_style="green", padding=(0, 2),
        ))
    else:
        print(f"\n  {'Metric':<42} {'Value':<22} What it Measures")
        print(f"  {'─'*42} {'─'*22} {'─'*34}")
        for label, val, note in rows:
            print(f"  {label:<42} {val:<22} {note}")
        print(f"\n{'═'*70}")
        print(f"  ✔  Complete.  {len(ledger)} transactions analysed.")
        print("═"*70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FL Blockchain Performance Metrics")
    parser.add_argument("--ledger", default="blockchain/ledger.json",
                        help="Path to ledger.json  (default: blockchain/ledger.json)")
    args = parser.parse_args()

    if HAS_RICH:
        with console.status("[bold cyan]Loading ledger and computing metrics…[/bold cyan]"):
            ledger  = load_ledger(args.ledger)
            metrics = compute_metrics(ledger)
    else:
        print("Loading ledger and computing metrics…")
        ledger  = load_ledger(args.ledger)
        metrics = compute_metrics(ledger)

    display_metrics(metrics, args.ledger, ledger)


if __name__ == "__main__":
    main()
