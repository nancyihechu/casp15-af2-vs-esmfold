"""
Quick ESMFold test on CASP15 target T1104 (117 residues, FM).
Run this on a GPU node, inside the esmfold conda env.

Output:
  - T1104_esmfold.pdb  (predicted structure, pLDDT stored in the B-factor column)
  - prints mean pLDDT, pTM and run time
"""

import time
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

TARGET = "T1104"
SEQ = (
    "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISIS"
    "AIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"
)

# 1. Check the GPU
assert torch.cuda.is_available(), "No GPU found. Are you on a GPU node?"
print("GPU:", torch.cuda.get_device_name(0))

# 2. Load the model (weights come from HF_HOME, no download needed)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model = model.cuda().eval()
model.esm = model.esm.half()  # language model in half precision: saves GPU memory
torch.backends.cuda.matmul.allow_tf32 = True
print(f"Model loaded in {time.time() - t0:.1f} s")

# 3. Fold the sequence and time it
torch.cuda.synchronize()
t1 = time.time()
with torch.no_grad():
    output = model.infer([SEQ])
torch.cuda.synchronize()
fold_time = time.time() - t1

# 4. Save the structure as a PDB file
pdb_text = model.output_to_pdb(output)[0]
out_file = f"{TARGET}_esmfold.pdb"
with open(out_file, "w") as f:
    f.write(pdb_text)

# 5. Report confidence
plddt_ca = output["plddt"][0, :, 1].float().cpu()  # index 1 = C-alpha atom
if plddt_ca.max() <= 1.0:                           # some versions use a 0-1 scale
    plddt_ca = plddt_ca * 100
ptm = float(output["ptm"].reshape(-1)[0]) if "ptm" in output else float("nan")

print(f"Target:      {TARGET} ({len(SEQ)} residues)")
print(f"Fold time:   {fold_time:.1f} s")
print(f"Mean pLDDT:  {plddt_ca.mean():.1f}")
print(f"pTM:         {ptm:.3f}")
print(f"Saved:       {out_file}")
