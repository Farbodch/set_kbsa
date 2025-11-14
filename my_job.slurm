#!/bin/bash

#SBATCH --output=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.out
#SBATCH --error=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.err
#SBATCH --job-name="cdr_sim"

#SBATCH --time=01:00:00
#SBATCH --clusters=inter
#SBATCH --partition=teramem_inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE

#SBATCH --mail-user=f.chamanian@tum.de
#SBATCH --mail-type=all


module load slurm_setup
source /dss/dsshome1/02/ge94nas3/.conda_init
conda activate set_kbsa

python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run.py