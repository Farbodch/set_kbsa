#!/bin/bash

CONDA_ENV_NAME="set_kbsa"
MPI_PROCS=8
SCRIPT=run_parallel_vecSob.py
NUM_OF_DATAPOINTS_ARR=(100 1000 10000)
ARRAY_LENGTH=3
NUM_OF_MESHPOINTS=1024
NUM_OF_MESHSTEPS=1024
NUM_OF_EXPERIMENTS=1
COMPUTE_STATISTICAL_FLAG=1

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "Error: Conda environment failed to activate. Exiting."
    exit 1
fi
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

for (( i=0; i<ARRAY_LENGTH; i++ ))
do
    RUN_NUM_OUTER=$((i + 1))
    CURRENT_VALUE="${NUM_OF_DATAPOINTS_ARR[$i]}"
    echo "Starting ${RUN_NUM_OUTER}/${ARRAY_LENGTH} with ${CURRENT_VALUE} points."
    for (( j = 0; j<NUM_OF_EXPERIMENTS; j++ ))
    do   
        RUN_NUM_INNER=$((j + 1))
        echo "${RUN_NUM_OUTER}: Starting experiment ${RUN_NUM_INNER}/${NUM_OF_EXPERIMENTS}."
        "$CONDA_PREFIX/bin/mpirun" \
            -n ${MPI_PROCS} python ${SCRIPT} --N "${CURRENT_VALUE}" --H "${NUM_OF_MESHPOINTS}" --S "${COMPUTE_STATISTICAL_FLAG}" --M "${NUM_OF_MESHSTEPS}"
        echo "${RUN_NUM_OUTER}: Experiment ${RUN_NUM_INNER}/${NUM_OF_EXPERIMENTS} completed."
        echo "----------------------------------------"
    done

    echo "Run ${RUN_NUM_INNER}/${NUM_OF_EXPERIMENTS} completed."
    echo "----------------------------------------"
done
conda deactivate
echo "All experiments finished!"