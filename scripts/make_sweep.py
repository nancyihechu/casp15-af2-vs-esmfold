"""
Make the MSA depth sweep from each target's full MSA.

Depth N = N sequences in total, counting the query. "full" = every sequence.
The query (first record) and any header lines (starting with '#') are always kept.
The other sequences are put in one random order per target (numpy seed from the
config, 0), and depth N keeps the first N-1 of them. So smaller depths are subsets
of larger ones. If N is larger than the MSA, the file equals the full MSA.

Writes results/<target>/af2/depth_<N>/<target>.a3m and results/tables/msa_sweep.tsv.
Reads only config/targets.yaml and results/<target>/af2/msa/<target>.a3m.
"""

import argparse
from pathlib import Path

import numpy as np

from project import load_config, read_sequence


def read_a3m(path):
    headers, records = [], []
    for line in path.read_text().replace("\x00", "").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            headers.append(line)
        elif line.startswith(">"):
            records.append([line, ""])
        else:
            records[-1][1] += line.strip()
    return headers, records


def build_sweep(cfg, tid):
    """Return (headers, {depth: records}) for one target, exactly as written to disk."""
    headers, records = read_a3m(cfg["results_dir"] / tid / f"af2/msa/{tid}.a3m")
    query, others = records[0], records[1:]
    if query[1] != read_sequence(cfg, tid):
        raise SystemExit(f"{tid}: first MSA record is not the target sequence")
    order = np.random.default_rng(cfg["random_seed"]).permutation(len(others))
    sweep = {}
    for depth in cfg["msa_depths"]:
        n_others = len(others) if depth == "full" else min(int(depth) - 1, len(others))
        sweep[str(depth)] = [query] + [others[i] for i in sorted(order[:n_others])]  # keep MSA order
    return headers, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    cfg = load_config(ap.parse_args().config)
    rows = ["target\tdepth\tn_sequences\tfull_msa_size\tcapped_at_full"]
    for t in cfg["targets"]:
        tid = t["id"]
        headers, sweep = build_sweep(cfg, tid)
        full_size = len(sweep["full"])
        for depth, chosen in sweep.items():
            out = cfg["results_dir"] / tid / f"af2/depth_{depth}/{tid}.a3m"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(headers + [f"{h}\n{s}" for h, s in chosen]) + "\n")
            capped = depth != "full" and int(depth) > full_size
            rows.append(f"{tid}\t{depth}\t{len(chosen)}\t{full_size}\t{capped}")
            print(f"{tid} depth_{depth}: {len(chosen)} sequences" + ("  (= full MSA)" if capped else ""))
    cfg["tables_dir"].mkdir(parents=True, exist_ok=True)
    (cfg["tables_dir"] / "msa_sweep.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
