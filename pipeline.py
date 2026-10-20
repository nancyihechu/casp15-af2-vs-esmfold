"""Run the project one step at a time: python pipeline.py <step>

Steps (filled in during later stages):
  collect, msa, sweep, af2, esmfold, perplexity, evaluate, analyze, report
"""
import sys

STEPS = ["collect", "msa", "sweep", "af2", "esmfold", "perplexity", "evaluate", "analyze", "report"]

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        sys.exit(f"Usage: python pipeline.py <{'|'.join(STEPS)}>")
    sys.exit(f"Step '{sys.argv[1]}' is not implemented yet.")
