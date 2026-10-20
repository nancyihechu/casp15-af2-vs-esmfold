"""
Shared helpers: load a target config and work out where its files live.

A config may set fasta_dir, natives_dir and results_dir (paths relative to the repo).
Defaults are data/fasta, data/natives and results, i.e. the main study.
config/positive_control.yaml points them elsewhere so the control stays separate.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config/targets.yaml"


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    cfg["fasta_dir"] = ROOT / cfg.get("fasta_dir", "data/fasta")
    cfg["natives_dir"] = ROOT / cfg.get("natives_dir", "data/natives")
    cfg["results_dir"] = ROOT / cfg.get("results_dir", "results")
    cfg["tables_dir"] = cfg["results_dir"] / "tables"
    return cfg


def read_sequence(cfg, tid):
    text = (cfg["fasta_dir"] / f"{tid}.fasta").read_text()
    return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
