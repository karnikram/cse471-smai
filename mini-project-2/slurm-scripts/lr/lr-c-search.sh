#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python lr.py --search_c > ../slurm-scripts/lr/lr-c-search.out
