#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_reservation
#SBATCH --reservation haicu_stefan
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --ntasks=1

source ~/.bashrc
conda activate rl_opt
python purejaxrl/evolve_ppo_atari.py

