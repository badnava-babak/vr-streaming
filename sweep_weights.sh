#!/usr/bin/env bash
# sweep_weights.sh  [N_JOBS]
# Grid-search w0,w1,w2 ∈ {0,0.1,…,10}.  Uses GNU parallel if N_JOBS>1.

# ── edit these paths ───────────────────────────────────────────────
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
CSV_LOG="results/metrics.csv"
POLICY="Optimal"
FIXED_ARGS=""                                  # e.g. "--env-config conf.yaml"
# ───────────────────────────────────────────────────────────────────

JOBS=${1:-1}                 # default serial
SEED=0                       # will ++ inside loop

run_cmd() {
  local w0=$1 w1=$2 w2=$3 seed=$4
  python "$PY_SCRIPT" \
         --policy "$POLICY" \
         --seed  "$seed" \
         --weights "$w0" "$w1" "$w2" \
         --csv-log "$CSV_LOG" \
         $FIXED_ARGS
}

export -f run_cmd            # needed for GNU parallel
export PY_SCRIPT POLICY CSV_LOG FIXED_ARGS

if [[ $JOBS -gt 1 ]]; then
  command -v parallel >/dev/null 2>&1 || {
    echo "GNU parallel not installed." >&2; exit 1; }
fi

for w0 in $(seq 0 0.1 10); do
  for w1 in $(seq 0 0.1 10); do
    for w2 in $(seq 0 0.1 10); do
      if [[ $JOBS -gt 1 ]]; then
        echo "$w0 $w1 $w2 $SEED"
      else
        # serial execution
        run_cmd "$w0" "$w1" "$w2" "$SEED"
      fi
      ((SEED++))
    done
  done
done | if [[ $JOBS -gt 1 ]]; then
        parallel -j "$JOBS" --colsep ' ' run_cmd {1} {2} {3} {4}
      else
        cat >/dev/null   # serial path already executed inline
      fi
