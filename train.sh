#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
 
# Load modules for MPI and other parallel libraries
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
 
# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER

cd /scratch-shared/$USER

srun python -u $HOME/master_thesis/train.py \
  --slr 0.001 --warmup_epoch 3 --c 0.2 --train_metrics \
  --seed 0 --num_epochs 100 --id _run_1 > output.txt
  
# # Copy output back from scratch to the directory from where the job was submitted
cp -r output.txt $HOME/master_thesis