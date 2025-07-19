#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem 100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Anaconda3/2023.09-0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_2


echo $(pwd)

python analysis_outputs.py