#!/bin/bash

# activate test_env in Aanaconda
export PATH=/home/dong/anaconda3/bin:/home/dong/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env


python trainer.py --model resnet50 --batch_size 64 --lr 0.00005 --epochs 100 --scheduler no --pretrained no
# [END] large-batch training (64,32,16,128)

# low-precision training
# discriminative lr
# lr-decay 
# label-smoothing
# knowledge distillation
