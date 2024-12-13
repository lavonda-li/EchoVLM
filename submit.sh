#!/bin/bash
#SBATCH --job-name=test_data
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G


module purge
module load python/3.12
python3 ${HOME}/EchoVLM/process_captions.py

