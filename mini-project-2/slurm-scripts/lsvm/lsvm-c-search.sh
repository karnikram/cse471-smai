#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 6
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ./../../code/
python lsvm.py --search_c > ./../slurm-scripts/lsvm/lsvm-c-search.out
