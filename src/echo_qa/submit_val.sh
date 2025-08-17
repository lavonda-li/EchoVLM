#!/bin/bash
#SBATCH --job-name=val_data
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module purge
module load python/3.12

# Set up environment
export PYTHONPATH="${HOME}/EchoPrime/src:${PYTHONPATH}"

# Run the processing
python3 -m echo_qa.cli --data_str val --batch_size 100 --start_idx 6600