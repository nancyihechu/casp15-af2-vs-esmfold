#!/bin/bash
# Run the whole pipeline with Claude Code as the agent (claude -p).
# The agent may run only `pipeline.py` commands (same idea as Project 2). It runs every step,
# reads each step's own check, keeps waiting while a Slurm job runs, retries a failed step
# once, and finishes with `pipeline.py report`, which writes results/report.md.
#
# Usage (on the login node: the collect and msa steps need internet):
#   ./run_agent.sh                                   # main study (config/targets.yaml)
#   ./run_agent.sh config/positive_control.yaml      # another target set
set -eo pipefail
cd "$(dirname "$0")"

PY=/cluster/pixstor/dkhf3-lab/users/nancy/envs/p3_tools/bin/python
PIPE="$PY pipeline.py"
CONFIG=${1:-}
CFG_ARG=${CONFIG:+ --config $CONFIG}
RESULTS=$($PY -c "import sys; sys.path.insert(0, 'scripts'); from project import load_config; \
print(load_config(sys.argv[1] if len(sys.argv) > 1 else None)['results_dir'])" $CONFIG)
LOG_DIR="$RESULTS/agent_logs"
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG_DIR"

command -v claude >/dev/null || { echo "claude (Claude Code) is not on PATH"; exit 1; }
[ -n "$SLURM_JOB_ID" ] && echo "Warning: run this on the login node; collect and msa need internet."
# A nested run must not inherit the settings of a Claude Code session it is started from
unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_SSE_PORT

PROMPT="You are running a protein structure prediction pipeline (CASP15 FM targets,
AlphaFold2 via ColabFold vs ESMFold). You may only run commands that start with:
  $PIPE
Run these steps in this order, one command per step:
  collect, msa, sweep, af2, esmfold, perplexity, evaluate, analyze
as: $PIPE <step>$CFG_ARG

Each command checks its own inputs and outputs and ends with one line:
  [OK] <step>: ...             the step is done; go to the next step
  [STILL_RUNNING] <step>: ...  (exit code 3) a Slurm job is still queued or running. This is not a
                               failure: run exactly the same command again, as many times as needed.
  [FAILED] <step>: ...         retry the same command once. If it fails a second time, stop: do not
                               run the later steps, go straight to the report.
Never add --force or --force-submit. Do not try any other command or tool.

At the end, always run:
  $PIPE report$CFG_ARG --notes \"<your notes>\"
where your notes are 2-6 plain sentences: which steps passed, which were retried and why,
which failed (quote the [FAILED] message), and anything in the [OK] messages that looks odd.
Then reply with a short summary of the run."

echo "Agent log: $LOG_DIR/agent_$STAMP.jsonl"
claude -p "$PROMPT" --output-format stream-json --verbose --max-turns 150 \
  --allowedTools "Bash($PIPE:*)" > "$LOG_DIR/agent_$STAMP.jsonl" 2> "$LOG_DIR/agent_$STAMP.err" || true

# Print the agent's final answer and the step status
$PY - "$LOG_DIR/agent_$STAMP.jsonl" <<'EOF'
import json, sys
final = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
result = [e for e in final if e.get("type") == "result"]
print("\n=== Agent summary ===")
print(result[-1].get("result", "(no result)") if result else "(the agent did not finish; see the .err log)")
EOF
$PIPE status$CFG_ARG
echo "Report: $RESULTS/report.md"
