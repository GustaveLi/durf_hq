#!/bin/bash
#SBATCH --nodes=1
#SBATCH -c 10
##SBATCH --time=1-12:00:00
#SBATCH --mem=360gb
#SBATCH --job-name="DIMRED_kPCA"
#SBATCH --mail-type=END
#SBATCH --mail-user=hl4212@nyu.edu
#SBATCH --output=%j.o
#SBATCH --error=%j.e
#SBATCH --partition=argon
#SBATCH -w argon22
#SBATCH --qos=argon

python /xspace/hl4212/durf_hq/projects/Gustave_Li/Main_program/Dimensionality_reduction.py
