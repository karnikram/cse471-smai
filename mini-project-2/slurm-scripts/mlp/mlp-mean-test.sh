#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 20
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python mlp.py --mean_sub --best > ../slurm-scripts/mlp/mlp-mean-test.out
