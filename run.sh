#!/bin/bash

# activate test_env in Aanaconda
# export PATH=/home/dong/anaconda3/bin:/home/dong/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_jh


# 1 [END] large-batch training (64,32,16,128)
# 2 [END] cos-lr-decay 
# 3 [END] Transfer Learning

# 4 [ING] Knowledge Distillation - Finding Teacher Model
# 

# making teacher model
# python trainer_transfer_learning.py --model resnet152 --batch_size 128 --lr 1e-2 --epochs 100 --scheduler yes --pretrained yes --transfer_learning_scheme 1
# python trainer_transfer_learning.py --model resnet152 --batch_size 128 --lr 1e-2 --epochs 100 --scheduler yes --pretrained yes --transfer_learning_scheme 2
# python trainer_transfer_learning.py --model resnet152 --batch_size 128 --lr 1e-2 --epochs 100 --scheduler yes --pretrained yes --transfer_learning_scheme 3 # on process


# 5 label-smoothing
# ssh -p 1011 root@165.246.38.201
# cd workspace/CNN_work
# conda activate test_jh