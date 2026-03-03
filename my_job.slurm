#!/bin/bash

#SBATCH --output=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.out
#SBATCH --error=/dss/dsshome1/02/ge94nas3/projects/set_kbsa/slurm_out/%x.%j.%N.err
#SBATCH --job-name="cdr_hsic_explicit_FEM_experiments"

#SBATCH --time=08:00:00
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=2
#SBATCH --export=NONE

#SBATCH --mail-user=f.chamanian@tum.de
#SBATCH --mail-type=all


module load slurm_setup
module load mpi
# source /dss/dsshome1/02/ge94nas3/.conda_init
source /dss/lrzsys/sys/spack/release/24.4.0/opt/x86_64/miniconda3/24.7.1-gcc-t6x7erm/etc/profile.d/conda.sh
conda activate set_kbsa

#-Data Generation
#   |
#   |---------cdr
#   |   |-----vecSob:
            # mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/data_generation/run_parallel_cdr_vecSob_experiments.py --N 5000             
#   |   |-----hsic:
            # mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/data_generation/run_parallel_cdr_hsic_experiments.py --N 5000
#   |---------diffusion_1d:
#   |   |-----vecSob:
          mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/data_generation/run_parallel_1d_diffusion_vecSob_experiments.py --N 10000
#   |   |-----hsic:
            # mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/data_generation/run_parallel_1d_diffusion_hsic_experiments.py --N 5000
#-----------------

#-Calculate Sensitivity Indicies
#   |
#   |---------vecSob:
        # mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/run_parallel_vecSob.py --N 1000 --M 1024 --H 1024 --S 0
#   |---------hsic:
        # mpiexec -n $SLURM_NTASKS python /dss/dsshome1/02/ge94nas3/projects/set_kbsa/run/run_parallel_hsic.py --N 10000
# -------------------------------
