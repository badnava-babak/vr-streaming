#!/bin/bash

#source /scratch/b502b586/venv/sdxl/bin/activate

CSV_LOG="/home/b502b586/workspaces/vr-streaming/results/ppg-exp"
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
FIXED_ARGS="--video-id 2 --verbose True --num-users 5 --num_episodes 1 --edge-proc-speed 12.e9 --load-model True"



run_cmd() {
  local w0=$1 w1=$2 w2=$3 seed=$4 policy=$5
  python "$PY_SCRIPT" \
         --policy "$policy" \
         --seed  "$seed" \
         --weights "$w0" "$w1" "$w2" \
         --csv-log "$CSV_LOG" \
         $FIXED_ARGS
}


w0=0.35
w1=0.85
w2=0.15
seed=42
# echo "These are the parameters for this job: $PARAMS --- $s -- $CSV_LOG"
run_cmd $w0 $w1 $w2 $seed "Optimal"
run_cmd $w0 $w1 $w2 $seed "PPG"
run_cmd $w0 $w1 $w2 $seed "CPPG"
#run_cmd $w0 $w1 $w2 $seed "EGreedy"

