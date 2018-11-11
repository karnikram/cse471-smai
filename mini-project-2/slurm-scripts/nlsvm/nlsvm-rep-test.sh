#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd ../../code
python nlsvm.py --scaling --best > ../slurm-scripts/nlsvm/nlsvm-scaling-test.out
python nlsvm.py --mean_sub --best > ../slurm-scripts/nlsvm/nlsvm-mean-test.out
python nlsvm.py --pca --best > ../slurm-scripts/nlsvm/nlsvm-pca-test.out
python nlsvm.py --lda --best > ../slurm-scripts/nlsvm/nlsvm-lda-test.out
