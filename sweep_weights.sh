#!/usr/bin/env bash
# sweep_weights.sh  [N_JOBS]
# Grid-search w0,w1,w2 ∈ {0,0.1,…,10}.  Uses GNU parallel if N_JOBS>1.

# ── edit these paths ───────────────────────────────────────────────
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
CSV_LOG="results/exp-f-5u"
POLICY="Optimal"
FIXED_ARGS="--video-id 2 --elastic True --num-users 5 --edge-proc-speed 12.e9"
# ───────────────────────────────────────────────────────────────────

STEP=${1:-0.1}          # first arg = step, else 0.1
JOBS=${2:-1}                 # second arg = parallelism, else 1
SEED=42                       # will ++ inside loop

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

# ---------------- build the grid & launch ----------------------
# Produce lines "w0 w1 w2 seed" that satisfy w0+w1+w2=1
generate_grid() {
  for w0 in $(seq 0 "$STEP" 1); do # psnr weight
    for w1 in $(seq 0 "$STEP" 1); do # stall time weight
      for w2 in $(seq 0 "$STEP" 1); do # energy consumption weight
      # compute w2 with awk (floating-point safe)
#      w1=$(awk -v a="$w0" -v b="$w2" 'BEGIN{printf "%.6f", 4-a-b}')
      # keep only settings where w2 ≥ 0  (allow tiny rounding error)
#      if awk -v c="$w1" 'BEGIN{exit(c<-1e-12)}'; then
        printf "%s %s %s %s\n" "$w0" "$w1" "$w2" "$SEED"
#        ((SEED++))
#      fi
      done
    done
  done
}

if [[ $JOBS -gt 1 ]]; then
  # parallel branch
  generate_grid | parallel -j "$JOBS" --colsep ' ' run_cmd {1} {2} {3} {4}
else
  # serial branch
  while read -r w0 w1 w2 seed; do
    run_cmd "$w0" "$w1" "$w2" "$seed"
  done < <(generate_grid)
fi

#head -n 1 -q  w0_0.15_w1_0.15_w2_0.15/stats.csv > exp.csv
#tail -n 1 -q  w0*/stats.csv >> exp.csv
