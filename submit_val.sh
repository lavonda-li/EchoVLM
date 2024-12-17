#!/bin/bash
#SBATCH --job-name=val_data
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G


module purge
module load python/3.12
python3 ${HOME}/EchoVLM/process_captions.py --data_str val --batch_size 100 --start_idx 6600