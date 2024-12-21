#!/bin/bash
#SBATCH --job-name=train_data
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G


module purge
module load python/3.12
python3 ${HOME}/EchoVLM/process_captions.py --data_str train --batch_size 1000 --start_idx 20000