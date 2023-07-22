#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --job-name=stc_2
 
module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

mkdir -p /scratch-shared/$USER
cp $HOME/master_thesis/train.py /scratch-shared/$USER
cd /scratch-shared/$USER

srun python -u $HOME/master_thesis/train.py \
  --slr 0.0001 \
  --c 0.0025 \
  --train_metrics \
  --seed 2.0 \
  --num_epochs 60 \
  --id _stochastic_2 \
  --train_stochastic \
  --save_state \
  > out_stochastic_2.txt

cp -r  out_stochastic_2.txt $HOME/master_thesis/output_files