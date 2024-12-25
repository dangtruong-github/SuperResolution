#!/bin/bash

#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J data
#SBATCH -p gpu --gpus 1
#SBATCH -A r00043
#SBATCH --mem=240G

module load python/gpu/3.10.10
module load miniconda
conda activate /N/slate/tnn3/TruongChu/.env/srgan

export PYTHONPATH="${PYTHONPATH}: /N/slate/tnn3/TruongChu/PyTorch-GAN/"

cd /N/slate/tnn3/TruongChu/PyTorch-GAN/implementations/srgan
python srgan.py