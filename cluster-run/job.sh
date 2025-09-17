#!/bin/bash
#SBATCH -p intel
#SBATCH -c 2
#SBATCH --mem=2GB
#SBATCH -t 10:00:00
#SBATCH -J weight_sweeping
#SBATCH -e logs/%A_%a.out
#SBATCH -o out/%A_%a.out


SLURM_ARRAY_TASK_ID=2
NUM_SIM=1000

START_IDX=$((SLURM_ARRAY_TASK_ID * NUM_SIM))
END_IDX=$(((SLURM_ARRAY_TASK_ID+1) * NUM_SIM))


#PARAMS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" params.txt)

STEP=0.01
source /scratch/b502b586/venv/sdxl/bin/activate

CSV_LOG="/home/b502b586/vr-streaming/results/exp-15/"
PY_SCRIPT="$(realpath ./run_task_offloading.py)"       # ABSOLUTE path now
POLICY="Optimal"
FIXED_ARGS=""                                  # e.g. "--env-config conf.yaml"

calc_weights() {
    local job_index=$1
    local step_size=$2

    # Number of possible values for each weight
    local num_steps=$(awk -v s="$step_size" 'BEGIN {print int(1/s)}')

    # Map job_index into i, j, k coordinates
    local i=$(( job_index / (num_steps * num_steps) ))
    local j=$(( (job_index / num_steps) % num_steps ))
    local k=$(( job_index % num_steps ))

    # Convert to actual weights (avoid 0 by adding 1 to index)
    local w0=$(awk -v idx=$i -v s="$step_size" 'BEGIN {print (idx+1)*s}')
    local w1=$(awk -v idx=$j -v s="$step_size" 'BEGIN {print (idx+1)*s}')
    local w2=$(awk -v idx=$k -v s="$step_size" 'BEGIN {print (idx+1)*s}')

    echo "$w0 $w1 $w2"
}

run_cmd() {
  local w0=$1 w1=$2 w2=$3 seed=$4
  python "$PY_SCRIPT" \
         --policy "$POLICY" \
         --seed  "$seed" \
         --weights "$w0" "$w1" "$w2" \
         --csv-log "$CSV_LOG" \
         $FIXED_ARGS
}



# Example usage:
# calc_weights "$SLURM_ARRAY_TASK_ID" 0.1

for s in $(seq "$START_IDX" 1 "$END_IDX"); do
  PARAMS=$(calc_weights $s $STEP)

  # echo "These are the parameters for this job: $PARAMS --- $s -- $CSV_LOG"
  run_cmd $PARAMS $s
done
