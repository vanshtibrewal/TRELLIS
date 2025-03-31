#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres gpu:h100:1   
#SBATCH --partition=gpu
#SBATCH -J "3D-VQ-VAE"   # job name
#SBATCH --mail-user=vtibrewa@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PYTHONUNBUFFERED=1
module load cuda/11.8.0-gcc-11.3.1-nlhqhb5

eval "$(conda shell.bash hook)"

if nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "V100"; then
    echo "V100 detected, activating V100 environment"
    conda activate trellis_v100
elif nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "H100"; then
    echo "H100 detected, activating H100 environment"
    conda activate trellis
elif nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -q "P100"; then
    echo "P100 detected, activating P100 environment"
    conda activate trellis_p100
else
    echo "Unknown GPU type, trying V100 environment"
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader
    conda activate trellis_v100
fi

python train.py \
  --config configs/vae/slat_vqvae_enc_dec_gs_swin8_B_64l8_fp16.json \
  --data_dir datasets/ObjaSubset \
  --auto_retry 0 \
  --output_dir outputs/frozen_longer_bigger
