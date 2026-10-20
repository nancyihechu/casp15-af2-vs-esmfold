"""
Fold every target with ESMFold (single sequence, no MSA). Run on a GPU node, in envs/esmfold,
with HF_HOME pointing to the local Hugging Face cache and HF_HUB_OFFLINE=1.

Writes, per target, to results/<target>/esmfold/:
  <target>_esmfold.pdb   predicted structure (pLDDT in the B-factor column)
  scores.json            per-residue pLDDT (C-alpha), pTM, PAE matrix
  summary.json           mean pLDDT, pTM, fold time (model loading not counted)
Reads only config/targets.yaml and data/fasta/.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from project import load_config, read_sequence


def fold(model, seq):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model.infer([seq])
    torch.cuda.synchronize()
    return output, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    cfg = load_config(ap.parse_args().config)
    assert torch.cuda.is_available(), "No GPU found. Are you on a GPU node?"
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu)

    t0 = time.time()
    AutoTokenizer.from_pretrained("facebook/esmfold_v1")  # checks the cache is complete
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()  # language model in half precision: saves GPU memory
    torch.backends.cuda.matmul.allow_tf32 = True
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f} s")

    # Warm-up fold (not timed), so CUDA start-up cost is not charged to the first target
    fold(model, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ")

    for t in cfg["targets"]:
        tid = t["id"]
        seq = read_sequence(cfg, tid)
        output, fold_time = fold(model, seq)

        out_dir = cfg["results_dir"] / tid / "esmfold"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_file = out_dir / f"{tid}_esmfold.pdb"
        pdb_file.write_text(model.output_to_pdb(output)[0])

        plddt = output["plddt"][0, :, 1].float().cpu()  # index 1 = C-alpha atom
        if plddt.max() <= 1.0:                          # some versions use a 0-1 scale
            plddt = plddt * 100
        ptm = float(output["ptm"].reshape(-1)[0])
        pae = output["predicted_aligned_error"][0].float().cpu()
        (out_dir / "scores.json").write_text(json.dumps({
            "plddt": [round(x, 2) for x in plddt.tolist()],
            "ptm": ptm,
            "pae": [[round(x, 2) for x in row] for row in pae.tolist()],
            "max_pae": float(pae.max())}))
        summary = {"target": tid, "length": len(seq), "pdb": pdb_file.name,
                   "mean_plddt": float(plddt.mean()), "ptm": ptm,
                   "fold_time_s": round(fold_time, 3), "model_load_time_s": round(load_time, 2),
                   "gpu": gpu, "timing_note": "fold_time_s = model.infer() only, after one warm-up fold"}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{tid} ({len(seq)} aa): fold {fold_time:.2f} s, mean pLDDT {plddt.mean():.1f}, "
              f"pTM {ptm:.3f}", flush=True)


if __name__ == "__main__":
    main()
