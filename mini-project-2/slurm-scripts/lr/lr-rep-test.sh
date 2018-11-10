#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python lr.py --scaling --best > ../slurm-scripts/lr/lr-scaling-test.out
python lr.py --mean_sub --best > ../slurm-scripts/lr/lr-mean-test.out
python lr.py --pca --best > ../slurm-scripts/lr/lr-pca-test.out
python lr.py --lda --best > ../slurm-scripts/lr/lr-lda-test.out
