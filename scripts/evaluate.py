"""
Evaluate step: the only step that reads the natives.

Every prediction (AF2 top model per depth, ESMFold) is trimmed to the FM domain (CASP
numbering) and scored against the native: TM-score, GDT-TS (0-100), RMSD (TMscore) and
all-atom lDDT (OpenStructure). Also records the model's own confidence (mean pLDDT over
the domain residues; pTM as reported for the whole chain) and runtime.

Timing columns:
  fold_time_s      AF2: fold_time_no_compile_s (5 x median of models 2-5, no JAX compile);
                   ESMFold: model.infer() time after a warm-up
  fold_time_raw_s  AF2: sum of the 5 per-model times incl. compile; ESMFold: same as fold_time_s
  time_per_model_s fold_time_s / number of models (5 for AF2, 1 for ESMFold)
  total_time_s     AF2: msa_time_s + fold_time_s; ESMFold: fold_time_s

Also writes results/<target>/superposition/ (native domain, AF2 full-MSA model and ESMFold
model, both moved onto the native with TMscore's superposition) for Figure 4, and, when
results/<target>/af2_all_models/ exists (reruns that keep all 5 AF2 models), scores every
model there into results/tables/af2_all_models.csv.

Needs results/tables/msa_depths.csv (scripts/msa_stats.py).
Writes results/tables/all_runs.csv and results/<target>/eval/ (trimmed domain models).
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys

import numpy as np

import scoring
from project import load_config, read_sequence

COLUMNS = ["target", "method", "msa_depth", "n_sequences", "is_duplicate_depth", "neff",
           "neff_per_L", "tm_score", "gdt_ts", "rmsd", "lddt", "mean_plddt", "ptm",
           "msa_time_s", "fold_time_s", "fold_time_raw_s", "time_per_model_s", "total_time_s",
           "n_residues_scored"]
RANGES = {"tm_score": (0, 1), "gdt_ts": (0, 100), "rmsd": (0, 100), "lddt": (0, 1),
          "mean_plddt": (0, 100), "ptm": (0, 1), "fold_time_s": (0, 3600)}


def domain_plddt(plddt, start, end):
    return float(np.mean(plddt[start - 1:end]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    cfg = load_config(ap.parse_args().config)
    with open(cfg["tables_dir"] / "msa_depths.csv") as f:
        msa = {(r["target"], r["msa_depth"]): r for r in csv.DictReader(f)}

    rows, extra_rows = [], []
    for t in cfg["targets"]:
        tid, (start, end) = t["id"], t["domain"]
        seq = read_sequence(cfg, tid)
        native = cfg["natives_dir"] / f"{tid}.pdb"
        tdir = cfg["results_dir"] / tid
        eval_dir = tdir / "eval"
        msa_time = json.loads((tdir / "af2/msa/msa_time.json").read_text())["msa_time_s"]

        sup_dir = tdir / "superposition"
        sup_dir.mkdir(exist_ok=True)
        shutil.copy(native, sup_dir / "native.pdb")

        for depth in map(str, cfg["msa_depths"]):
            run = tdir / f"af2/depth_{depth}"
            summary = json.loads((run / "summary.json").read_text())
            top = summary["models"][0]
            assert top["rank"] == 1
            plddt = json.loads((run / top["scores_file"]).read_text())["plddt"]
            pdb = run / summary["top_model_pdb"]
            eval_dir_run = eval_dir / f"af2_depth_{depth}"
            s = scoring.score(pdb, native, seq, start, end, eval_dir_run)
            m = msa[(tid, depth)]
            fold = summary["fold_time_no_compile_s"]
            rows.append({"target": tid, "method": "AF2", "msa_depth": depth,
                         "n_sequences": int(m["n_sequences"]),
                         "is_duplicate_depth": m["is_duplicate_depth"] == "True",
                         "neff": float(m["neff"]), "neff_per_L": float(m["neff_per_L"]),
                         **{k: s[k] for k in ("tm_score", "gdt_ts", "rmsd", "lddt")},
                         "mean_plddt": domain_plddt(plddt, start, end), "ptm": summary["ptm"],
                         "msa_time_s": msa_time, "fold_time_s": fold,
                         "fold_time_raw_s": summary["fold_time_s"],
                         "time_per_model_s": fold / len(summary["models"]),
                         "total_time_s": msa_time + fold, "n_residues_scored": s["n_common"]})
            if depth == "full":
                scoring.superpose(s["trimmed_pdb"], native, sup_dir / "af2_full.pdb")
            print(f"{tid} AF2 depth_{depth}: TM {s['tm_score']:.3f}", flush=True)

        esm_dir = tdir / "esmfold"
        summary = json.loads((esm_dir / "summary.json").read_text())
        plddt = json.loads((esm_dir / "scores.json").read_text())["plddt"]
        s = scoring.score(esm_dir / summary["pdb"], native, seq, start, end, eval_dir / "esmfold")
        fold = summary["fold_time_s"]
        rows.append({"target": tid, "method": "ESMFold", "msa_depth": "", "n_sequences": "",
                     "is_duplicate_depth": False, "neff": "", "neff_per_L": "",
                     **{k: s[k] for k in ("tm_score", "gdt_ts", "rmsd", "lddt")},
                     "mean_plddt": domain_plddt(plddt, start, end), "ptm": summary["ptm"],
                     "msa_time_s": "", "fold_time_s": fold, "fold_time_raw_s": fold,
                     "time_per_model_s": fold, "total_time_s": fold,
                     "n_residues_scored": s["n_common"]})
        scoring.superpose(s["trimmed_pdb"], native, sup_dir / "esmfold.pdb")
        print(f"{tid} ESMFold: TM {s['tm_score']:.3f}", flush=True)

        for run in sorted((tdir / "af2_all_models").glob("depth_*")):
            gpu = json.loads((run / "summary.json").read_text()).get("gpu", "")
            for pdb in sorted(run.glob(f"{tid}_unrelaxed_rank_*.pdb")):
                rank = int(re.search(r"rank_(\d+)", pdb.name).group(1))
                model = re.search(r"(alphafold2_ptm_model_\d)", pdb.name).group(1)
                scores = json.loads(next(run.glob(f"{tid}_scores_rank_{rank:03d}_*.json")).read_text())
                s = scoring.score(pdb, native, seq, start, end, eval_dir / f"all_models_{run.name}")
                extra_rows.append({"target": tid, "msa_depth": run.name.replace("depth_", ""),
                                   "rank": rank, "model": model,
                                   **{k: s[k] for k in ("tm_score", "gdt_ts", "rmsd", "lddt")},
                                   "mean_plddt": domain_plddt(scores["plddt"], start, end),
                                   "ptm": scores["ptm"], "gpu": gpu})
            print(f"{tid} all 5 AF2 models scored for {run.name}", flush=True)

    # sanity checks: every number in a sensible range
    problems = [f"{r['target']} {r['method']} {r['msa_depth']}: {k}={r[k]}"
                for r in rows for k, (lo, hi) in RANGES.items()
                if not (isinstance(r[k], (int, float)) and math.isfinite(r[k]) and lo <= r[k] <= hi)]

    out = cfg["tables_dir"] / "all_runs.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print(f"Wrote {out} ({len(rows)} rows)")
    if extra_rows:
        out = cfg["tables_dir"] / "af2_all_models.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(extra_rows[0]))
            w.writeheader()
            for r in extra_rows:
                w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        print(f"Wrote {out} ({len(extra_rows)} rows)")
    if problems:
        sys.exit("Out-of-range values:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
