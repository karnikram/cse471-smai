#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
#python lsvm.py --scaling --best > ../slurm-scripts/lsvm/lsvm-scaling-test.out
#python lsvm.py --mean_sub --best > ../slurm-scripts/lsvm/lsvm-mean-test.out
python lsvm.py --pca --best > ../slurm-scripts/lsvm/lsvm-pca-test.out
#python lsvm.py --lda --best > ../slurm-scripts/lsvm/lsvm-lda-test.out
