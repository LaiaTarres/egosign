#!/bin/bash
#SBATCH --time=5-11:00:00
#SBATCH --mem 16G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Anaconda3/2023.09-0
module load FFmpeg/4.3.2-GCCcore-10.2.0
eval "$(conda shell.bash hook)"
conda activate SLTopicDetection_3

#Això surt al terminal, com que estem fent random search, volem limitar-ho perquè no sigui infinit.
#For the transformer CLS:
#wandb agent /SLTopicDetection_sweep/a0ke0xyf --count 20
#wandb agent /SLTopicDetection_sweep/8zhheppz --count 20

#For the LSTM:
#wandb agent /SLTopicDetection_sweep/tlpv2oab --count 20

#For the perceiverIO:
wandb agent /SLTopicDetection_sweep/lur8yfyp --count 20 #This is the one with the body as well.