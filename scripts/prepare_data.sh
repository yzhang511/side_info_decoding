#!/bin/bash

#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 2
#SBATCH --mem 100000
#SBATCH -t 0-1
#SBATCH --export=ALL

. ~/.bashrc
echo $TMPDIR
conda activate decoding

session_id=${1}

cd ..
python src/allen_visual_behavior_neuropixels/prepare_data.py \
    --session_id $session_id \
    --data_dir /scratch/bdtg/yzhang39/allen/datasets/ 

conda deactivate

