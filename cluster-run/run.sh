#!/bin/bash

#NUMPARAMS=$(cat params.txt | wc -l)

sbatch --array=0-10000 job.sh
sbatch --array=10001-20000 job.sh
