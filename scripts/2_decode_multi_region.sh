#!/bin/bash

#SBATCH --account=stats             
#SBATCH --job-name="re_decode"
#SBATCH --output="re_decode.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=50G       
#SBATCH --time=2-0:00              

module load anaconda

. ~/.bashrc
echo $TMPDIR
conda activate ibl_repro_ephys
cd /burg/stats/users/yz4123/neural_decoding

python src/2_decode_multi_region.py --target block --query_region PO LP DG CA1 VISa --base_path /burg/stats/users/yz4123/Downloads 

cd script

conda deactivate
