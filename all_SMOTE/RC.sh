#!/usr/bin/env bash

#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH -J RC-all-SMOTE
#SBATCH -p veryhimem
#SBATCH -c 8
#SBATCH -N 1     
#SBATCH -o %x-%j.out
#SBATCH --mail-user=yiyan.wu@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

module load python3

python /cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all_SMOTE/RegionalClassifier.py
