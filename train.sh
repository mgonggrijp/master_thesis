#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --job-name=cw_s_1
 
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER
cd /scratch-shared/$USER

srun python -u $HOME/master_thesis/train.py \
  --slr 0.0001 \
  --c 0.2 \
  --train_metrics \
  --seed 1.0 \
  --num_epochs 60 \
  --id _clsw_sq_1 \
  --save_state \
  --weights class_based \
  > out_clsw_sq_1.txt

cp -r  out_clsw_sq_1.txt $HOME/master_thesis/output_files