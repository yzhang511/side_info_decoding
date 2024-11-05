#!/bin/bash

#SBATCH --account=stats             
#SBATCH --job-name="re_cache"
#SBATCH --output="re_cache.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=2-0:00              

module load anaconda

. ~/.bashrc
echo $TMPDIR
conda activate ibl_repro_ephys
cd /burg/stats/users/yz4123/neural_decoding/src

python 2_decode_multi_region.py --target choice --query_region PO LP DG CA1 VISa --base_path /burg/stats/users/yz4123/Downloads 

cd script

conda deactivate