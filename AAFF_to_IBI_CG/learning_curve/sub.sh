#!/bin/bash
#SBATCH --job-name=RAND
#SBATCH --mem=4096
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --error=Wham.stderr
#SBATCH --output=Wham.stdout
#SBATCH --time=10:00:00




cd $SLURM_SUBMIT_DIR
python do_all.py

