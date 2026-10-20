"""
Get one MSA per target from the ColabFold MMseqs2 server (colabfold_batch --msa-only).
No templates are requested. Run on the login node (needs internet).

Writes, per target:
  results/<target>/af2/msa/<target>.a3m       the full MSA
  results/<target>/af2/msa/msa_time.json      wall-clock time of the server search
Reads only config/targets.yaml and data/fasta/.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COLABFOLD = "/cluster/pixstor/dkhf3-lab/users/nancy/envs/colabfold/bin/colabfold_batch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", help="default: all targets in the config")
    ap.add_argument("--force", action="store_true", help="redo targets that already have an MSA")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/targets.yaml").read_text())
    ids = args.targets or [t["id"] for t in cfg["targets"]]

    for tid in ids:
        out_dir = ROOT / f"results/{tid}/af2/msa"
        a3m = out_dir / f"{tid}.a3m"
        if a3m.exists() and not args.force:
            print(f"{tid}: MSA exists, skipping")
            continue
        work = out_dir / "colabfold_msa_only"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        t0 = time.time()
        proc = subprocess.run([COLABFOLD, "--msa-only", str(ROOT / f"data/fasta/{tid}.fasta"),
                               str(work)], capture_output=True, text=True)
        elapsed = time.time() - t0
        (out_dir / "msa_only.log").write_text(proc.stdout + proc.stderr)
        produced = work / f"{tid}.a3m"
        if proc.returncode != 0 or not produced.exists():
            raise SystemExit(f"{tid}: MSA search failed (see {out_dir / 'msa_only.log'})")
        shutil.copy(produced, a3m)
        shutil.rmtree(work)
        n_seqs = sum(1 for line in a3m.read_text().splitlines() if line.startswith(">"))
        (out_dir / "msa_time.json").write_text(json.dumps(
            {"target": tid, "msa_time_s": round(elapsed, 2), "n_sequences": n_seqs,
             "server": "ColabFold MMseqs2 (UniRef + environmental), no templates"}, indent=2))
        print(f"{tid}: {n_seqs} sequences, MSA search {elapsed:.1f} s")


if __name__ == "__main__":
    main()
