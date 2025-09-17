#!/bin/bash

#NUMPARAMS=$(cat params.txt | wc -l)

BATCH_SIZE=1000
TOTAL_NUM_RUNS=999997
# TOTAL_NUM_RUNS=1000

for s in $(seq -1 "$BATCH_SIZE" "$TOTAL_NUM_RUNS"); do
    end=$((s+BATCH_SIZE))
    start=$((s+1))
    sbatch --array=${start}-${end} cluster-run/job.sh
    # echo "$start" "$end"
    sleep 30s
done

# sbatch --array=10001-20000 job.sh
