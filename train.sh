#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=10:30:00
#SBATCH --job-name=run_3
 
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER
cd /scratch-shared/$USER

srun python -u $HOME/master_thesis/train.py \
  --slr 0.0001 --c 0.2 --train_metrics --seed 3.0 --num_epochs 60 --id _run_3 \
  --save_state > outputs_run_3.txt

cp -r  outputs_run_3.txt $HOME/master_thesis/output_files