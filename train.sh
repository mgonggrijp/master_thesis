#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
 
# Load modules for MPI and other parallel libraries
module load 2022
module load Anaconda3/2022.05
 
# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER
  
cd /scratch-shared/$USER
# Execute the program in parallel on ntasks cores

source hyperbolic
srun python -m $HOME/master_thesis/train.py  output
  
# Copy output back from scratch to the directory from where the job was submitted
cp -r output $SLURM_SUBMIT_DIR