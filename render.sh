#!/bin/bash
#SBATCH --job-name=blender_render
#SBATCH --array=0-7                   # Create 8 independent jobs (indices 0 through 7)
#SBATCH --nodes=1                   # Each array job gets 1 node
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=8           # Adjust based on your script’s --max_workers
#SBATCH --mem=128G                  # Adjust based on Blender scenes and object size
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out       # %A is the master job ID, %a is the array task ID
#SBATCH --error=%x-%A_%a.err

export PYTHONUNBUFFERED=TRUE
conda activate trellis
module load cuda/11.8.0-gcc-11.3.1-nlhqhb5

python dataset_toolkits/render.py ObjaverseXL \
    --output_dir datasets/ObjaSubset \
    --rank=$SLURM_ARRAY_TASK_ID \
    --world_size=$SLURM_ARRAY_TASK_COUNT
