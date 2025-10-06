#!/bin/bash

#source /scratch/b502b586/venv/sdxl/bin/activate

CSV_LOG="/home/b502b586/workspaces/vr-streaming/results/elasticity-exp"
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
FIXED_ARGS="--video-id 2 --save-model True --tensorboard False --verbose True --num-users 5 --num_episodes 2000 --edge-proc-speed 12.e9"



w0=0.35
w1=0.85
w2=0.15
policy="PPG"
seed=42

python "$PY_SCRIPT" --elastic False --elasticity-parameter 0 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS &
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 1 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 2 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS &
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 3 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 4 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS &
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 5 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS
#python "$PY_SCRIPT" --elastic False --elasticity-parameter 6 --policy "$policy" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS &
#python "$PY_SCRIPT" --elastic True --policy "PPG" --seed  "$seed" --weights "$w0" "$w1" "$w2" --csv-log "$CSV_LOG" $FIXED_ARGS