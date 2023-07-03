#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
 
# Load modules for MPI and other parallel libraries
module load 2022

# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER
  
cd /scratch-shared/$USER
# Execute the program in parallel on ntasks cores
 
srun $HOME/master_thesis/train.py $HOME/master_thesis/outputs.txt  
  
# Copy output back from scratch to the directory from where the job was submitted
cp -r $HOME/master_thesis/outputs.txt $SLURM_SUBMIT_DIR