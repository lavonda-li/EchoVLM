#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G


module purge
module load python/3.12
python3 ${HOME}/EchoVLM/process_captions.py

