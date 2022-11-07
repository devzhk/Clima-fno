#!/bin/sh

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=32   # number of threads
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "pyqg"   # job name
#SBATCH --no-requeue
#SBATCH --mail-user=zhaoyi@caltech.edu   # email address
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=1-3


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /home/zhaoyi/ml_function
python run_pyqg.py nx256beta${SLURM_ARRAY_TASK_ID}rek1p2.json

