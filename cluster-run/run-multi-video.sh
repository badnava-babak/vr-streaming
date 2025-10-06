#!/bin/bash

#source /scratch/b502b586/venv/sdxl/bin/activate

CSV_LOG="/home/b502b586/workspaces/vr-streaming/results/multi-video"
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
FIXED_ARGS="--random-video True --elastic True --save-model True --tensorboard False --verbose True --num-users 8 --num_episodes 5000 --edge-proc-speed 12.e9"



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
# echo "These are the parameters for this job: $PARAMS --- $s -- $CSV_LOG"
run_cmd $w0 $w1 $w2 42 "Optimal"
run_cmd $w0 $w1 $w2 42 "PPG"
run_cmd $w0 $w1 $w2 42 "CPPG"
#run_cmd $w0 $w1 $w2 42 "EGreedy"
#run_cmd $w0 $w1 $w2 42 "PPO"

