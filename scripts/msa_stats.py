"""
MSA statistics and input checks for every AF2 run.

For each results/<target>/af2/depth_<N>/<target>.a3m (the file AF2 actually read;
ColabFold rewrites it without the extra copies of the query: the MSA server puts the
query at the top of both its UniRef and its environmental section) this:
  1. counts the sequences (query included),
  2. computes Neff and Neff/L: each sequence gets weight 1 / (number of sequences,
     itself included, with >= 80% identity to it); Neff = sum of weights; L = query length.
     Identity = fraction of identical columns over the L aligned columns (a3m insertions,
     the lowercase letters, removed; a gap matching a gap counts as identical),
  3. checks the file is the planned subsample (make_sweep.build_sweep, rebuilt in memory)
     with only the extra query copies removed,
  4. checks log.txt: AF2 loaded that many sequences ("Setting max_seq=..") and there is
     no sign of an MSA server query,
  5. marks duplicate depth runs: runs of one target with exactly the same input sequences.
     The kept run is "full" if it is in the group, otherwise the largest depth.

Writes results/tables/msa_depths.csv. Exits with an error if any check fails.
"""

import argparse
import csv
import re
import sys

import numpy as np

from make_sweep import build_sweep, read_a3m
from project import load_config

IDENTITY = 0.8
SERVER_SIGNS = re.compile(r"mmseqs2|api\.colabfold|SUBMIT|PENDING|RUNNING:|COMPLETE:|Server",
                          re.IGNORECASE)


def aligned(seq):
    return "".join(c for c in seq if not c.islower())


def neff(seqs):
    arr = np.array([np.frombuffer(s.encode(), dtype=np.uint8) for s in seqs])
    n, length = arr.shape
    counts = np.zeros(n)
    for i in range(0, n, 256):
        ident = (arr[i:i + 256, None, :] == arr[None, :, :]).mean(axis=2)
        counts[i:i + 256] = (ident >= IDENTITY).sum(axis=1)
    return float((1.0 / counts).sum())


def without_query_copies(records):
    """ColabFold drops later records with the query's header (">101"); identical hits stay."""
    query_header, query = records[0]
    return [query] + [s for h, s in records[1:] if h != query_header]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    cfg = load_config(ap.parse_args().config)
    rows, problems = [], []
    for t in cfg["targets"]:
        tid = t["id"]
        _, planned = build_sweep(cfg, tid)
        runs = {}
        for depth in map(str, cfg["msa_depths"]):
            run_dir = cfg["results_dir"] / tid / f"af2/depth_{depth}"
            _, records = read_a3m(run_dir / f"{tid}.a3m")
            seqs = [s for _, s in records]
            log = (run_dir / "log.txt").read_text()
            stdout = (run_dir / "colabfold_stdout.txt").read_text()

            m = re.search(r"Setting max_seq=(\d+), max_extra_seq=(\d+)", log)
            log_n = None
            if m:
                max_seq, max_extra = int(m.group(1)), int(m.group(2))
                log_n = max_seq + max_extra if max_seq == 512 else max_seq
            is_planned = seqs == without_query_copies(planned[depth])
            server = bool(SERVER_SIGNS.search(log) or SERVER_SIGNS.search(stdout))
            if log_n != len(seqs):
                problems.append(f"{tid} depth_{depth}: log says {log_n} sequences, file has {len(seqs)}")
            if not is_planned:
                problems.append(f"{tid} depth_{depth}: .a3m is not the planned subsample")
            if server:
                problems.append(f"{tid} depth_{depth}: log mentions the MSA server")

            n_eff = neff([aligned(s) for s in seqs])
            runs[depth] = tuple(seqs)
            rows.append({"target": tid, "msa_depth": depth, "n_sequences": len(seqs),
                         "planned_n_sequences": len(planned[depth]),
                         "neff": round(n_eff, 2), "neff_per_L": round(n_eff / len(seqs[0]), 4),
                         "log_n_sequences": log_n, "is_planned_subsample": is_planned,
                         "msa_server_query_in_log": server})

        # duplicate depths: identical input sequences within a target
        for row in (r for r in rows if r["target"] == tid):
            group = [d for d, s in runs.items() if s == runs[row["msa_depth"]]]
            keep = "full" if "full" in group else max(group, key=int)
            row["is_duplicate_depth"] = row["msa_depth"] != keep
            row["duplicate_of"] = keep if row["is_duplicate_depth"] else ""

    cfg["tables_dir"].mkdir(parents=True, exist_ok=True)
    out = cfg["tables_dir"] / "msa_depths.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} runs)")
    if problems:
        sys.exit("Checks failed:\n  " + "\n  ".join(problems))
    print("All checks passed: every AF2 run read its planned subsampled .a3m "
          "(extra query copies removed) and no run queried the MSA server.")


if __name__ == "__main__":
    main()
