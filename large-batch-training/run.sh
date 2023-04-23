#!/bin/bash

# Only for server conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate {env_name}


# large-batch training
python trainer_transfer_learning.py --model resnet34 --batch_size 64 --lr 1e-3 --epochs 100 --scheduler no --pretrained no --transfer_learning_scheme 3 # on process