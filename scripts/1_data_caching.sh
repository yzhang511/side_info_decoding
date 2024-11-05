#!/bin/bash

#SBATCH --account=stats             
#SBATCH --job-name="re_cache"
#SBATCH --output="re_cache.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=0-5:00              

module load anaconda

. ~/.bashrc
echo $TMPDIR
conda activate ibl_repro_ephys
cd /burg/stats/users/yz4123/neural_decoding/src
python 1_data_caching.py --base_path /burg/stats/users/yz4123/Downloads
conda deactivate
cd ../scripts