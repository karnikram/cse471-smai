#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 8
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python mlp.py --pca --best > ../slurm-scripts/mlp/mlp-pca-test.out
