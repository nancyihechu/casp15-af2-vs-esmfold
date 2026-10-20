"""
Scoring helpers (used only by the evaluate step; this is the only code that reads natives).

- trim_to_domain: keep only the FM-domain residues of a prediction (CASP numbering),
  after checking that the prediction's sequence matches the CASP target sequence.
- tmscore: TM-score, GDT-TS and RMSD from the Zhang lab TMscore program. It pairs
  residues by residue number and normalizes by the native (second file) length.
- lddt: all-atom lDDT from OpenStructure (`ost compare-structures`, run in an
  Apptainer container), with residues paired by residue number.

Run inside envs/p3_tools (TMscore is in its bin/).
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

OST_SIF = Path("/cluster/pixstor/dkhf3-lab/users/nancy/containers/ost.sif")
TMSCORE = shutil.which("TMscore", path=str(Path(sys.executable).parent)) or shutil.which("TMscore")


def trim_to_domain(pred_pdb, target_seq, start, end, out_pdb):
    """Write the domain residues [start, end] of pred_pdb to out_pdb (chain A).
    Prediction residue i must be target_seq[i-1]; raises ValueError otherwise."""
    s = PDBParser(QUIET=True).get_structure("p", pred_pdb)
    chains = list(s[0])
    if len(chains) != 1:
        raise ValueError(f"{pred_pdb}: expected 1 chain, found {len(chains)}")
    lines, serial, kept = [], 1, 0
    for res in chains[0]:
        num, name = res.id[1], res.get_resname()
        if protein_letters_3to1.get(name, "X") != target_seq[num - 1]:
            raise ValueError(f"{pred_pdb}: residue {name}{num} does not match target "
                             f"{target_seq[num - 1]}{num}")
        if not start <= num <= end:
            continue
        kept += 1
        for atom in res:
            x, y, z = atom.coord
            an = atom.get_name()
            field = f" {an:<3}" if len(an) < 4 else an
            lines.append(f"ATOM  {serial:5d} {field} {name:>3} A{num:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{atom.bfactor:6.2f}"
                         f"          {atom.element:>2}")
            serial += 1
    Path(out_pdb).write_text("\n".join(lines + ["TER", "END"]) + "\n")
    return kept


def tmscore(model_pdb, native_pdb):
    out = subprocess.run([TMSCORE, str(model_pdb), str(native_pdb)],
                         capture_output=True, text=True, check=True).stdout

    def grab(pattern):
        m = re.search(pattern, out)
        if not m:
            raise ValueError(f"TMscore output missing '{pattern}':\n{out}")
        return float(m.group(1))

    return {"tm_score": grab(r"TM-score\s*=\s*([\d.]+)"),
            "gdt_ts": 100 * grab(r"GDT-TS-score\s*=\s*([\d.]+)"),  # 0-100 scale, as in CASP
            "rmsd": grab(r"RMSD of\s+the common residues\s*=\s*([\d.]+)"),
            "n_common": int(grab(r"Number of residues in common\s*=\s*(\d+)"))}


def lddt(model_pdb, native_pdb):
    model_pdb, native_pdb = Path(model_pdb).resolve(), Path(native_pdb).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "ost.json"
        subprocess.run(
            ["apptainer", "exec", "--bind", ",".join(sorted({str(model_pdb.parent),
                                                            str(native_pdb.parent), tmp})),
             str(OST_SIF), "ost", "compare-structures",
             "-m", str(model_pdb), "-r", str(native_pdb), "-rna", "--lddt",
             "-o", str(out_json)],
            capture_output=True, text=True, check=True)
        result = json.loads(out_json.read_text())
    if result.get("status") != "SUCCESS" or result.get("lddt") is None:
        raise ValueError(f"OpenStructure lDDT failed: {result.get('traceback', result)}")
    return {"lddt": float(result["lddt"])}


def score(pred_pdb, native_pdb, target_seq, start, end, workdir):
    """Trim pred_pdb to the domain and score it against native_pdb."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    trimmed = workdir / (Path(pred_pdb).stem + "_domain.pdb")
    trim_to_domain(pred_pdb, target_seq, start, end, trimmed)
    return {**tmscore(trimmed, native_pdb), **lddt(trimmed, native_pdb)}
