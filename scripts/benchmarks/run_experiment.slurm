#!/bin/bash

#SBATCH -A imi@v100
#SBATCH --time=03:00:00
#SBATCH --input=none
#SBATCH --mail-user=gautier.hamon@inria.fr
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err



python train.py -config configs/OpenES/vector_recipes.yaml




