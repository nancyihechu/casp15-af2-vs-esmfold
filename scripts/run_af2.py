"""
Run AlphaFold2 (ColabFold, alphafold2_ptm weights) on every MSA depth of one target.
Templates OFF (no --templates), no relax (no --amber). Run inside a GPU job.

For each results/<target>/af2/depth_<N>/<target>.a3m this writes, in the same folder:
  <target>_unrelaxed_rank_001_*.pdb   top model (ranked by mean pLDDT)
  <target>_scores_rank_00k_*.json     ColabFold scores of all 5 models (pLDDT, pTM, PAE)
  log.txt                             ColabFold log (per-model timings)
  summary.json                        top model, per-model scores, fold time
Reads only config/targets.yaml and the .a3m files.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
COLABFOLD = "/cluster/pixstor/dkhf3-lab/users/nancy/envs/colabfold/bin/colabfold_batch"
WEIGHTS = "/cluster/pixstor/dkhf3-lab/users/nancy/colabfold_weights"
FLAGS = ["--model-type", "alphafold2_ptm", "--num-models", "5", "--num-recycle", "3",
         "--random-seed", "0", "--rank", "plddt", "--data", WEIGHTS]


def run_depth(tid, depth_dir):
    a3m = depth_dir / f"{tid}.a3m"
    # clear outputs from an earlier attempt, keep the input MSA
    for f in depth_dir.iterdir():
        if f.name != a3m.name:
            f.unlink()
    cmd = [COLABFOLD, str(a3m), str(depth_dir), *FLAGS]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    (depth_dir / "colabfold_stdout.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{tid} {depth_dir.name}: colabfold_batch failed, see colabfold_stdout.txt")

    log = (depth_dir / "log.txt").read_text()
    # e.g. "alphafold2_ptm_model_1_seed_000 took 25.3s (3 recycles)"
    model_times = {m: float(s) for m, s in re.findall(r"(alphafold2_ptm_model_\d_seed_\d+) took ([\d.]+)s", log)}
    models = []
    for f in sorted(depth_dir.glob(f"{tid}_scores_rank_*.json")):
        s = json.loads(f.read_text())
        name = re.search(r"(alphafold2_ptm_model_\d_seed_\d+)", f.name).group(1)
        rank = int(re.search(r"rank_(\d+)", f.name).group(1))
        models.append({"rank": rank, "model": name, "scores_file": f.name,
                       "mean_plddt": float(np.mean(s["plddt"])), "ptm": float(s["ptm"]),
                       "time_s": model_times.get(name)})
    if len(models) != 5:
        raise RuntimeError(f"{tid} {depth_dir.name}: expected 5 models, found {len(models)}")
    top = min(models, key=lambda m: m["rank"])
    top_pdb = next(depth_dir.glob(f"{tid}_unrelaxed_rank_001_*.pdb"))

    # keep the top PDB, all score files and the log; drop the rest (other PDBs, PNGs, duplicates)
    keep = {a3m.name, top_pdb.name, "log.txt", "colabfold_stdout.txt", "config.json"} | \
           {m["scores_file"] for m in models}
    for f in depth_dir.iterdir():
        if f.name not in keep:
            f.unlink()

    n_seqs = sum(1 for line in a3m.read_text().splitlines() if line.startswith(">"))
    summary = {"target": tid, "depth": depth_dir.name.replace("depth_", ""), "n_sequences": n_seqs,
               "top_model_pdb": top_pdb.name, "top_model": top["model"],
               "mean_plddt": top["mean_plddt"], "ptm": top["ptm"],
               "fold_time_s": round(sum(m["time_s"] for m in models), 2),
               # model_1 runs first and carries the one-off JAX compilation; replace its time
               # by the median of the other four to estimate folding time without compilation
               "fold_time_no_compile_s": round(5 * float(np.median(
                   [m["time_s"] for m in models if not m["model"].startswith("alphafold2_ptm_model_1_")])), 2),
               "colabfold_wall_time_s": round(wall, 2),
               "timing_note": "fold_time_s = sum of the 5 per-model times in log.txt "
                              "(model_1 includes JAX compilation); fold_time_no_compile_s = "
                              "5 x median time of models 2-5; colabfold_wall_time_s also "
                              "includes weight loading",
               "flags": FLAGS, "models": sorted(models, key=lambda m: m["rank"])}
    (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{tid} {depth_dir.name}: {n_seqs} seqs, top {top['model']} pLDDT {top['mean_plddt']:.1f} "
          f"pTM {top['ptm']:.3f}, fold {summary['fold_time_s']:.0f} s, wall {wall:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="target id")
    ap.add_argument("--index", type=int, help="target index in the config (Slurm array task id)")
    ap.add_argument("--depths", nargs="*", help="default: all depths in the config")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/targets.yaml").read_text())
    tid = args.target or cfg["targets"][args.index]["id"]
    depths = args.depths or [str(d) for d in cfg["msa_depths"]]
    failed = []
    for d in depths:
        try:
            run_depth(tid, ROOT / f"results/{tid}/af2/depth_{d}")
        except Exception as e:  # keep going with the other depths, report at the end
            print(f"[FAIL] {e}", flush=True)
            failed.append(d)
    if failed:
        raise SystemExit(f"{tid}: failed depths {failed}")


if __name__ == "__main__":
    main()
