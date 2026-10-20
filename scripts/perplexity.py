"""
ESM-2 pseudo-perplexity (facebook/esm2_t36_3B_UR50D) for every target.

Mask each position one at a time, take the model's probability of the true amino acid
at the masked position, then pseudo-perplexity = exp(mean negative log probability).
Computed twice: on the full CASP sequence, and on the FM-domain part alone
(the domain residues given to the model as their own sequence).
Run on a GPU node, in envs/esmfold, with HF_HUB_OFFLINE=1.

Writes results/<target>/perplexity.json and results/tables/perplexity.csv.
Reads only config/targets.yaml and data/fasta/.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

from project import load_config, read_sequence

MODEL = "facebook/esm2_t36_3B_UR50D"
BATCH = 16


def pseudo_perplexity(model, tokenizer, seq):
    ids = tokenizer(seq, return_tensors="pt")["input_ids"][0].cuda()  # <cls> seq <eos>
    positions = torch.arange(1, len(seq) + 1, device="cuda")
    nll = torch.empty(len(seq))
    for b in range(0, len(seq), BATCH):
        pos = positions[b:b + BATCH]
        batch = ids.repeat(len(pos), 1)
        batch[torch.arange(len(pos)), pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=batch).logits.float()
        logp = torch.log_softmax(logits[torch.arange(len(pos)), pos], dim=-1)
        nll[b:b + len(pos)] = -logp[torch.arange(len(pos)), ids[pos]].cpu()
    return math.exp(nll.mean().item()), nll.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    cfg = load_config(ap.parse_args().config)
    assert torch.cuda.is_available(), "No GPU found. Are you on a GPU node?"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForMaskedLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    rows = []
    for t in cfg["targets"]:
        tid, (start, end) = t["id"], t["domain"]
        seq = read_sequence(cfg, tid)
        t0 = time.time()
        ppl_full, nll_full = pseudo_perplexity(model, tokenizer, seq)
        ppl_dom, nll_dom = pseudo_perplexity(model, tokenizer, seq[start - 1:end])
        elapsed = time.time() - t0
        result = {"target": tid, "model": MODEL, "length": len(seq), "domain": [start, end],
                  "pseudo_perplexity_full": ppl_full, "pseudo_perplexity_domain": ppl_dom,
                  "time_s": round(elapsed, 2),
                  "per_residue_nll_full": [round(x, 4) for x in nll_full],
                  "per_residue_nll_domain": [round(x, 4) for x in nll_dom]}
        (cfg["results_dir"] / tid / "perplexity.json").write_text(json.dumps(result, indent=1))
        rows.append({k: result[k] for k in ("target", "length", "pseudo_perplexity_full",
                                            "pseudo_perplexity_domain", "time_s")})
        print(f"{tid}: pseudo-perplexity full {ppl_full:.3f}, domain {ppl_dom:.3f} ({elapsed:.1f} s)",
              flush=True)

    with open(cfg["tables_dir"] / "perplexity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
