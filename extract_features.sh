#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --array=0-7                   
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=16           
#SBATCH --mem=128G                  
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out       
#SBATCH --error=%x-%A_%a.err

export PYTHONUNBUFFERED=TRUE

eval "$(conda shell.bash hook)"
conda activate trellis

module load cuda/11.8.0-gcc-11.3.1-nlhqhb5

srun python dataset_toolkits/extract_feature.py \
    --output_dir datasets/ObjaSubset \
    --batch_size 16 \
    --rank $SLURM_ARRAY_TASK_ID \
    --world_size $SLURM_ARRAY_TASK_COUNT
