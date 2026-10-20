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

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


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


def main():
    cfg = yaml.safe_load((ROOT / "config/targets.yaml").read_text())
    rows = ["target\tdepth\tn_sequences\tfull_msa_size\tcapped_at_full"]
    for t in cfg["targets"]:
        tid = t["id"]
        headers, records = read_a3m(ROOT / f"results/{tid}/af2/msa/{tid}.a3m")
        query, others = records[0], records[1:]
        query_seq = (ROOT / f"data/fasta/{tid}.fasta").read_text().split("\n", 1)[1].replace("\n", "")
        if query[1] != query_seq:
            raise SystemExit(f"{tid}: first MSA record is not the target sequence")
        order = np.random.default_rng(cfg["random_seed"]).permutation(len(others))
        for depth in cfg["msa_depths"]:
            n_others = len(others) if depth == "full" else min(int(depth) - 1, len(others))
            chosen = [others[i] for i in sorted(order[:n_others])]  # keep original MSA order
            out = ROOT / f"results/{tid}/af2/depth_{depth}/{tid}.a3m"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(headers + [f"{h}\n{s}" for h, s in [query] + chosen]) + "\n")
            capped = depth != "full" and int(depth) - 1 > len(others)
            rows.append(f"{tid}\t{depth}\t{n_others + 1}\t{len(records)}\t{capped}")
            print(f"{tid} depth_{depth}: {n_others + 1} sequences" + ("  (= full MSA)" if capped else ""))
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/tables/msa_sweep.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
