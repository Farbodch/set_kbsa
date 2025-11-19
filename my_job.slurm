#!/bin/bash

#SBATCH --output=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.out
#SBATCH --error=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.err
#SBATCH --job-name="cdr_experiments"

#SBATCH --time=02:00:00
#SBATCH --clusters=inter
#SBATCH --partition=teramem_inter
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE

#SBATCH --mail-user=f.chamanian@tum.de
#SBATCH --mail-type=all


module load slurm_setup
module load mpi
source /dss/dsshome1/02/ge94nas3/.conda_init
conda activate set_kbsa

mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run_parallel_experiments.py