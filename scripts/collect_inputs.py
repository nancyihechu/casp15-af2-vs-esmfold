"""
Stage 1: collect the inputs.

1. Download the CASP15 target sequences and write data/fasta/<target>.fasta.
   Check that each length matches config/targets.yaml.
2. Get a native structure for each target, trimmed to the FM domain, in CASP numbering:
   a. first choice: the official CASP15 domain file (<target>-D1.pdb)
   b. fallback: the RCSB entry. Pick the chain that best matches the target sequence,
      align it to the target, renumber to CASP numbering, trim to the domain.
3. Check every native: residue numbers inside the domain range, and the residue at each
   position matches the target sequence.

Writes data/natives/<target>.pdb and data/natives/sources.tsv.
Run on the login node (needs internet), in envs/p3_tools.
"""

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

ROOT = Path(__file__).resolve().parent.parent
SEQ_URL = "https://predictioncenter.org/download_area/CASP15/sequences/casp15.seq.txt"
DOM_URL = ("https://predictioncenter.org/download_area/CASP15/targets/"
           "casp15.targets.TS-domains.public_12.20.2022.tar.gz")
RCSB_URL = "https://files.rcsb.org/download/{}.cif"
MIN_CHAIN_IDENTITY = 0.90  # below this, the PDB chain is probably not the target


def download(url, dest):
    if not dest.exists():
        print(f"Downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def read_casp_sequences(path):
    seqs, name = {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line.strip()
    return seqs


def one_letter(resname):
    resname = "MET" if resname == "MSE" else resname
    return protein_letters_3to1.get(resname.upper(), "X")


def protein_residues(chain):
    """Standard amino-acid residues (plus selenomethionine) with a CA atom."""
    out = []
    for res in chain:
        hetflag = res.id[0]
        if (hetflag == " " or res.get_resname() == "MSE") and "CA" in res:
            if one_letter(res.get_resname()) != "X":
                out.append(res)
    return out


def map_to_target(residues, target_seq):
    """Align chain sequence to target sequence. Return {residue index: CASP number}
    for identical aligned positions, plus identity over the aligned chain residues."""
    chain_seq = "".join(one_letter(r.get_resname()) for r in residues)
    aligner = Align.PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                    open_gap_score=-5, extend_gap_score=-0.5,
                                    end_gap_score=0)
    aln = aligner.align(chain_seq, target_seq)[0]
    mapping, aligned, identical = {}, 0, 0
    for (c0, c1), (t0, t1) in zip(*aln.aligned):
        for ci, ti in zip(range(c0, c1), range(t0, t1)):
            aligned += 1
            if chain_seq[ci] == target_seq[ti]:
                identical += 1
                mapping[ci] = ti + 1  # CASP numbering starts at 1
    identity = identical / aligned if aligned else 0.0
    return mapping, identity, len(chain_seq)


def write_pdb(residue_numbers, out_path, header):
    """residue_numbers: list of (Bio residue, new CASP number). Writes chain A, heavy atoms."""
    lines = [f"REMARK {header}"]
    serial = 1
    for res, num in residue_numbers:
        resname = "MET" if res.get_resname() == "MSE" else res.get_resname()
        for atom in res:
            if atom.element == "H":
                continue
            if atom.is_disordered():
                atom = atom.selected_child
            name = atom.get_name()
            if name == "SE" and res.get_resname() == "MSE":
                name, element = "SD", "S"
            else:
                element = atom.element
            x, y, z = atom.coord
            atom_field = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {atom_field} {resname:>3} A{num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{atom.bfactor:6.2f}          {element:>2}")
            serial += 1
    lines += ["TER", "END"]
    out_path.write_text("\n".join(lines) + "\n")


def native_from_casp(pdb_path, target_seq, start, end, out_path):
    s = PDBParser(QUIET=True).get_structure("n", pdb_path)
    chain = next(s[0].get_chains())
    pairs = [(r, r.id[1]) for r in protein_residues(chain) if start <= r.id[1] <= end]
    write_pdb(pairs, out_path, f"source CASP15 domain file {pdb_path.name}")
    return "CASP15 domain file", chain.id, None


def native_from_rcsb(cif_path, pdb_id, target_seq, start, end, out_path):
    s = MMCIFParser(QUIET=True).get_structure(pdb_id, cif_path)
    best = None
    for chain in s[0]:
        residues = protein_residues(chain)
        if len(residues) < 20:
            continue
        mapping, identity, n = map_to_target(residues, target_seq)
        score = (identity, len(mapping))
        if best is None or score > best[0]:
            best = (score, chain.id, residues, mapping)
    if best is None:
        raise RuntimeError(f"{pdb_id}: no protein chain found")
    (identity, n_mapped), chain_id, residues, mapping = best
    if identity < MIN_CHAIN_IDENTITY:
        raise RuntimeError(f"{pdb_id} chain {chain_id}: identity {identity:.2f} to the "
                           f"target is too low; this may be the wrong entry")
    pairs = [(residues[i], num) for i, num in sorted(mapping.items(), key=lambda kv: kv[1])
             if start <= num <= end]
    write_pdb(pairs, out_path, f"source RCSB {pdb_id} chain {chain_id}, "
                               f"renumbered to CASP numbering, trimmed to {start}-{end}")
    return f"RCSB {pdb_id}", chain_id, identity


def check_native(path, target_seq, start, end):
    s = PDBParser(QUIET=True).get_structure("n", path)
    nums, mismatches = [], []
    for res in s[0]["A"]:
        num = res.id[1]
        nums.append(num)
        if not (start <= num <= end):
            raise RuntimeError(f"{path.name}: residue {num} outside domain {start}-{end}")
        if one_letter(res.get_resname()) != target_seq[num - 1]:
            mismatches.append(f"{res.get_resname()}{num}")
    if mismatches:
        raise RuntimeError(f"{path.name}: sequence mismatch at {mismatches}")
    if len(nums) != len(set(nums)):
        raise RuntimeError(f"{path.name}: duplicate residue numbers")
    return len(nums), min(nums), max(nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config/targets.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    dl = ROOT / "data/downloads"
    seqs = read_casp_sequences(download(SEQ_URL, dl / "casp15.seq.txt"))
    dom_tar = download(DOM_URL, dl / "casp15.targets.TS-domains.public_12.20.2022.tar.gz")
    with tarfile.open(dom_tar) as tf:
        casp_files = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
        casp_bytes = {name: tf.extractfile(m).read() for name, m in casp_files.items()
                      if any(name == f"{t['id']}-D1.pdb" for t in cfg["targets"])}

    (ROOT / "data/fasta").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/natives").mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    for t in cfg["targets"]:
        tid, start, end = t["id"], *t["domain"]
        seq = seqs.get(tid)
        if seq is None:
            print(f"[FAIL] {tid}: not in the CASP15 sequence file"); ok = False; continue
        if len(seq) != t["length"]:
            print(f"[FAIL] {tid}: length {len(seq)} != expected {t['length']}"); ok = False
        (ROOT / f"data/fasta/{tid}.fasta").write_text(f">{tid}\n{seq}\n")

        out = ROOT / f"data/natives/{tid}.pdb"
        try:
            if f"{tid}-D1.pdb" in casp_bytes:
                raw = dl / "casp_domains" / f"{tid}-D1.pdb"
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(casp_bytes[f"{tid}-D1.pdb"])
                source, chain, ident = native_from_casp(raw, seq, start, end, out)
            else:
                cif = download(RCSB_URL.format(t["pdb"].upper()), dl / f"rcsb/{t['pdb']}.cif")
                source, chain, ident = native_from_rcsb(cif, t["pdb"], seq, start, end, out)
            n, lo, hi = check_native(out, seq, start, end)
        except RuntimeError as e:
            print(f"[FAIL] {tid}: {e}"); ok = False; continue
        dom_len = end - start + 1
        print(f"[OK] {tid}: seq len {len(seq)}, native from {source} chain {chain}, "
              f"{n}/{dom_len} domain residues present ({lo}-{hi})"
              + (f", chain identity {ident:.3f}" if ident is not None else ""))
        rows.append(f"{tid}\t{source}\t{chain}\t{start}-{end}\t{n}\t{dom_len}\t{lo}-{hi}")

    (ROOT / "data/natives/sources.tsv").write_text(
        "target\tsource\tchain\tdomain\tresidues_present\tdomain_length\tobserved_range\n"
        + "\n".join(rows) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
