"""Run the project one step at a time:  python pipeline.py <step> [--config FILE] [--force]

Steps, in order:
  collect     CASP15 sequences + FM-domain natives              (login node, internet)
  msa         one MSA per target from the ColabFold server      (login node, internet)
  sweep       MSA depth subsamples                              (login node)
  af2         ColabFold on every depth                          (Slurm GPU job, L40S)
  esmfold     ESMFold on every target                           (Slurm GPU job, L40S)
  perplexity  ESM-2 pseudo-perplexity                           (Slurm GPU job, L40S)
  evaluate    MSA checks + Neff, score every run vs the native  (login node)
  analyze     tables, figures, results block in README.md       (login node)
  report      results/report.md                                 (login node)
Also: python pipeline.py status

Every step checks that its inputs exist, skips work whose outputs are already complete
(unless --force), runs, then checks its outputs (files exist, numbers in a sensible range).
Exit codes: 0 = step OK, 1 = step failed (message says why), 3 = a Slurm job is still
running: run the same command again to keep waiting (this is not a failure).
Each call is logged to <results>/pipeline_log.jsonl.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from project import load_config  # noqa: E402

PY = "/cluster/pixstor/dkhf3-lab/users/nancy/envs/p3_tools/bin/python"
STEPS = ["collect", "msa", "sweep", "af2", "esmfold", "perplexity", "evaluate", "analyze", "report"]
STILL_RUNNING = 3
FIGURES = ["fig1_tm_vs_n_sequences.png", "fig1b_tm_vs_neff_per_L.png", "fig2_perplexity_vs_tm.png",
           "fig3_confidence_vs_accuracy.png", "fig4_structures.png", "fig5_pae.png",
           "fig6_accuracy_vs_speed.png", "fig7_perplexity_vs_msa_depth.png", "fig8_tm_vs_perplexity_neff.png"]


class StepFailed(Exception):
    pass


class StillRunning(Exception):
    pass


# ------------------------------------------------------------------ helpers

def require(paths, what):
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise StepFailed(f"missing {what}: {', '.join(missing[:5])}"
                         + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))


def in_range(value, lo, hi, what):
    if not (isinstance(value, (int, float)) and math.isfinite(value) and lo <= value <= hi):
        raise StepFailed(f"{what} = {value!r}, expected {lo}..{hi}")


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, text=True, capture_output=True)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    if len(out) > 25:
        print(f"... ({len(out) - 25} earlier lines not shown)", flush=True)
    print("\n".join(out[-25:]), flush=True)
    if proc.returncode != 0:
        raise StepFailed(f"{Path(str(cmd[1])).name} exited with code {proc.returncode}: "
                         + (out[-1] if out else "no output"))


def cfg_args(args):
    return ["--config", args.config] if args.config else []


def ids(cfg):
    return [t["id"] for t in cfg["targets"]]


def json_file(path):
    return json.loads(path.read_text())


# ------------------------------------------------------------ Slurm jobs

def slurm_step(cfg, args, name, sbatch_file, extra_sbatch=()):
    """Submit sbatch_file once, then wait for it (at most args.wait_minutes per call)."""
    jobs = cfg["results_dir"] / ".jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    record = jobs / f"{name}.json"
    job = json_file(record) if record.exists() else None
    if job is None or args.force_submit:
        env = f"ALL,CONFIG={args.config}" if args.config else "ALL"
        cmd = ["sbatch", "--parsable", f"--export={env}", *extra_sbatch, sbatch_file]
        out = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if out.returncode != 0:
            raise StepFailed(f"sbatch failed: {out.stderr.strip()}")
        job = {"job_id": out.stdout.strip().split(";")[0], "submitted": time.time()}
        record.write_text(json.dumps(job))
        print(f"Submitted Slurm job {job['job_id']} ({sbatch_file})", flush=True)
    jid = job["job_id"]
    deadline = time.time() + 60 * args.wait_minutes
    while subprocess.run(["squeue", "-h", "-j", jid], text=True, capture_output=True).stdout.strip():
        if time.time() > deadline:
            raise StillRunning(f"Slurm job {jid} is still queued or running; "
                               f"run `python pipeline.py {name}` again to keep waiting")
        time.sleep(30)
    states = subprocess.run(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "State"],
                            text=True, capture_output=True).stdout.split()
    record.unlink()
    print(f"Slurm job {jid} finished: {', '.join(sorted(set(states))) or 'unknown'}", flush=True)
    if any(s not in ("COMPLETED",) for s in states):
        raise StepFailed(f"Slurm job {jid} ended with {sorted(set(states))}; see slurm/logs/")


# ------------------------------------------------------------- output checks

def check_collect(cfg):
    for t in cfg["targets"]:
        fasta = cfg["fasta_dir"] / f"{t['id']}.fasta"
        native = cfg["natives_dir"] / f"{t['id']}.pdb"
        require([fasta, native], "inputs")
        seq = "".join(fasta.read_text().splitlines()[1:])
        if len(seq) != t["length"]:
            raise StepFailed(f"{t['id']}: sequence length {len(seq)} != {t['length']}")
        nums = {int(line[22:26]) for line in native.read_text().splitlines() if line.startswith("ATOM")}
        lo, hi = t["domain"]
        if not nums or min(nums) < lo or max(nums) > hi:
            raise StepFailed(f"{t['id']}: native residues outside domain {lo}-{hi}")
    require([cfg["natives_dir"] / "sources.tsv"], "native source table")
    return f"{len(cfg['targets'])} sequences and natives OK"


def check_msa(cfg):
    out = []
    for tid in ids(cfg):
        d = cfg["results_dir"] / tid / "af2/msa"
        require([d / f"{tid}.a3m", d / "msa_time.json"], "MSA files")
        info = json_file(d / "msa_time.json")
        in_range(info["msa_time_s"], 0, 3600, f"{tid} msa_time_s")
        in_range(info["n_sequences"], 1, 10**7, f"{tid} n_sequences")
        out.append(f"{tid}:{info['n_sequences']}")
    return ("MSAs OK (raw server sequence counts, query listed twice; not what AF2 used, see evaluate: "
            + ", ".join(out) + ")")


def a3m_paths(cfg):
    return [cfg["results_dir"] / tid / f"af2/depth_{d}/{tid}.a3m" for tid in ids(cfg) for d in cfg["msa_depths"]]


def af2_done(cfg, tid):
    for d in cfg["msa_depths"]:
        f = cfg["results_dir"] / tid / f"af2/depth_{d}/summary.json"
        if not f.exists():
            return False
        s = json_file(f)
        if len(s["models"]) != 5 or not (f.parent / s["top_model_pdb"]).exists():
            return False
    return True


def check_af2(cfg):
    n = 0
    for tid in ids(cfg):
        if not af2_done(cfg, tid):
            raise StepFailed(f"{tid}: AF2 outputs incomplete")
        for d in cfg["msa_depths"]:
            s = json_file(cfg["results_dir"] / tid / f"af2/depth_{d}/summary.json")
            in_range(s["mean_plddt"], 0, 100, f"{tid} depth_{d} mean_plddt")
            in_range(s["ptm"], 0, 1, f"{tid} depth_{d} ptm")
            in_range(s["fold_time_no_compile_s"], 0, 3600, f"{tid} depth_{d} fold time")
            n += 1
    return f"{n} AF2 runs OK"


def check_esmfold(cfg):
    for tid in ids(cfg):
        d = cfg["results_dir"] / tid / "esmfold"
        require([d / "summary.json", d / "scores.json", d / f"{tid}_esmfold.pdb"], "ESMFold outputs")
        s = json_file(d / "summary.json")
        in_range(s["mean_plddt"], 0, 100, f"{tid} ESMFold mean_plddt")
        in_range(s["ptm"], 0, 1, f"{tid} ESMFold ptm")
        in_range(s["fold_time_s"], 0, 600, f"{tid} ESMFold fold_time_s")
        if "L40S" not in s["gpu"]:
            raise StepFailed(f"{tid}: ESMFold ran on {s['gpu']}, not an L40S")
    return f"{len(cfg['targets'])} ESMFold runs OK"


def check_perplexity(cfg):
    path = cfg["tables_dir"] / "perplexity.csv"
    require([path], "perplexity table")
    rows = list(csv.DictReader(path.open()))
    if sorted(r["target"] for r in rows) != sorted(ids(cfg)):
        raise StepFailed("perplexity.csv does not list exactly the config targets")
    for r in rows:  # a 20-letter alphabet: uniform guessing gives 20, a perfect model 1
        in_range(float(r["pseudo_perplexity_full"]), 1, 30, f"{r['target']} perplexity")
        in_range(float(r["pseudo_perplexity_domain"]), 1, 30, f"{r['target']} domain perplexity")
    return f"{len(rows)} perplexities OK"


def check_evaluate(cfg):
    path = cfg["tables_dir"] / "all_runs.csv"
    require([path, cfg["tables_dir"] / "msa_depths.csv"], "evaluation tables")
    rows = list(csv.DictReader(path.open()))
    expected = len(cfg["targets"]) * (len(cfg["msa_depths"]) + 1)
    if len(rows) != expected:
        raise StepFailed(f"all_runs.csv has {len(rows)} rows, expected {expected}")
    for r in rows:
        tag = f"{r['target']} {r['method']} {r['msa_depth']}"
        in_range(float(r["tm_score"]), 0, 1, f"{tag} tm_score")
        in_range(float(r["lddt"]), 0, 1, f"{tag} lddt")
        in_range(float(r["gdt_ts"]), 0, 100, f"{tag} gdt_ts")
        in_range(float(r["total_time_s"]), 0, 7200, f"{tag} total_time_s")
    used = {r["target"]: r["n_sequences"] for r in csv.DictReader((cfg["tables_dir"] / "msa_depths.csv").open())
            if r["msa_depth"] == "full"}
    return (f"all_runs.csv: {len(rows)} rows OK; full-MSA sequences AF2 used (msa_depths.csv): "
            + ", ".join(f"{t}:{used[t]}" for t in ids(cfg)))


def check_analyze(cfg, since=None):
    figs = [cfg["results_dir"] / "figures" / f for f in FIGURES]
    require(figs + [cfg["results_dir"] / "summary.md"], "analysis outputs")
    if since is not None:  # every output must come from this run, not an earlier one
        old = [f.name for f in figs + [cfg["results_dir"] / "summary.md"] if f.stat().st_mtime < since]
        if old:
            raise StepFailed(f"not rewritten by this run (older files left over): {old}")
    small = [f.name for f in figs if f.stat().st_size < 10_000]
    if small:
        raise StepFailed(f"figures look empty: {small}")
    extra = sorted(f.name for f in (cfg["results_dir"] / "figures").glob("figS*.png"))
    return f"{len(figs)} main figures" + (f" + {len(extra)} supplementary" if extra else "") + " and summary.md OK"


# ---------------------------------------------------------------- the steps

def step_collect(cfg, args):
    run([PY, "scripts/collect_inputs.py", *cfg_args(args)])
    return check_collect(cfg)


def step_msa(cfg, args):
    check_collect(cfg)
    run([PY, "scripts/get_msa.py", *cfg_args(args), *(["--force"] if args.force else [])])
    return check_msa(cfg)


def step_sweep(cfg, args):
    check_msa(cfg)
    if all(af2_done(cfg, tid) for tid in ids(cfg)) and not args.force:
        # ColabFold rewrites the depth .a3m files when it runs; rewriting them now would
        # make them differ from what AF2 actually used.
        require(a3m_paths(cfg), "depth MSA files")
        return "skipped: AF2 already ran on these depth files (use --force, then af2 --force)"
    run([PY, "scripts/make_sweep.py", *cfg_args(args)])
    require(a3m_paths(cfg), "depth MSA files")
    return f"{len(a3m_paths(cfg))} depth MSA files OK"


def step_af2(cfg, args):
    require(a3m_paths(cfg), "depth MSA files (run sweep first)")
    todo = [i for i, tid in enumerate(ids(cfg)) if args.force or not af2_done(cfg, tid)]
    if todo:
        slurm_step(cfg, args, "af2", "slurm/af2_array.sbatch", [f"--array={','.join(map(str, todo))}"])
    return check_af2(cfg) + ("" if todo else " (already complete)")


def step_esmfold(cfg, args):
    check_collect(cfg)
    try:
        if args.force:
            raise StepFailed("forced")
        return check_esmfold(cfg) + " (already complete)"
    except StepFailed:
        slurm_step(cfg, args, "esmfold", "slurm/esmfold.sbatch")
    return check_esmfold(cfg)


def step_perplexity(cfg, args):
    check_collect(cfg)
    try:
        if args.force:
            raise StepFailed("forced")
        return check_perplexity(cfg) + " (already complete)"
    except StepFailed:
        slurm_step(cfg, args, "perplexity", "slurm/perplexity.sbatch")
    return check_perplexity(cfg)


def step_evaluate(cfg, args):
    check_collect(cfg)
    check_af2(cfg)
    check_esmfold(cfg)
    run([PY, "scripts/msa_stats.py", *cfg_args(args)])
    run([PY, "scripts/evaluate.py", *cfg_args(args)])
    return check_evaluate(cfg)


def step_analyze(cfg, args):
    check_evaluate(cfg)
    check_perplexity(cfg)
    start = time.time()
    run([PY, "scripts/analyze.py", *cfg_args(args)])
    return check_analyze(cfg, since=start)


def last_status(cfg):
    log = cfg["results_dir"] / "pipeline_log.jsonl"
    status = {}
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            status[entry["step"]] = entry
    return status


def step_report(cfg, args):
    check_analyze(cfg)
    status = last_status(cfg)
    lines = ["# Pipeline report", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python pipeline.py report`.", "",
             "## Step status (last run of each step)", "",
             "| step | status | when | time (s) | message |", "|---|---|---|---|---|"]
    for step in STEPS[:-1]:
        e = status.get(step)
        lines.append(f"| {step} | {e['status']} | {e['time']} | {e['seconds']:.0f} | {e['message']} |"
                     if e else f"| {step} | not run | | | |")
    if args.notes:
        lines += ["", "## Notes from the agent", "", args.notes]
    lines += ["", "## Results", "", (cfg["results_dir"] / "summary.md").read_text(),
              "Figures: `results/figures/`. Full method and limits: `README.md`."]
    out = cfg["results_dir"] / "report.md"
    out.write_text("\n".join(lines) + "\n")
    not_ok = [s for s in STEPS[:-1] if status.get(s, {}).get("status") != "ok"]
    if not_ok:
        raise StepFailed(f"report written, but these steps are not ok: {not_ok}")
    return f"wrote {out.relative_to(ROOT)}"


def step_status(cfg, args):
    status = last_status(cfg)
    for step in STEPS:
        e = status.get(step)
        print(f"{step:11s} {e['status']:13s} {e['time']}  {e['message']}" if e else f"{step:11s} not run")
    return None


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["status"])
    ap.add_argument("--config", help="target config (default: config/targets.yaml)")
    ap.add_argument("--force", action="store_true", help="redo the step even if its outputs are complete")
    ap.add_argument("--force-submit", action="store_true", help="submit a new Slurm job even if one is recorded")
    ap.add_argument("--wait-minutes", type=float, default=9, help="max wait per call for a Slurm job")
    ap.add_argument("--notes", default="", help="report: text to include (e.g. retries)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    func = globals()[f"step_{args.step}"]
    if args.step == "status":
        func(cfg, args)
        return

    t0 = time.time()
    try:
        message, status, code = func(cfg, args), "ok", 0
    except StillRunning as e:
        message, status, code = str(e), "still_running", STILL_RUNNING
    except StepFailed as e:
        message, status, code = str(e), "failed", 1
    entry = {"step": args.step, "status": status, "message": message,
             "seconds": round(time.time() - t0, 1), "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
             "config": args.config or "config/targets.yaml"}
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    with open(cfg["results_dir"] / "pipeline_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{status.upper()}] {args.step}: {message}")
    sys.exit(code)


if __name__ == "__main__":
    main()
