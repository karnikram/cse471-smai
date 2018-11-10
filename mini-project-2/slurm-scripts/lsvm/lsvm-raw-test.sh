#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 4
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python lsvm.py --best > ../slurm-scripts/lsvm/lsvm-raw-test.out

