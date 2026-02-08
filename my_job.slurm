#!/bin/bash

#SBATCH --output=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/vecSob_data/%x.%j.%N.out
#SBATCH --error=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/vecSob_data/%x.%j.%N.err
#SBATCH --job-name="cdr_vecSob_experiments"

#SBATCH --time=08:00:00
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks=37
#SBATCH --cpus-per-task=3
#SBATCH --export=NONE

#SBATCH --mail-user=f.chamanian@tum.de
#SBATCH --mail-type=all


module load slurm_setup
module load mpi
# source /dss/dsshome1/02/ge94nas3/.conda_init
source /dss/lrzsys/sys/spack/release/24.4.0/opt/x86_64/miniconda3/24.7.1-gcc-t6x7erm/etc/profile.d/conda.sh
conda activate set_kbsa

# mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run_parallel_cdr_vecSob_experiments.py --N 4000
mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run_parallel_vecSob.py

# mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run_parallel_cdr_experiments.py --N 5000
# mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run_parallel_hsic.py
