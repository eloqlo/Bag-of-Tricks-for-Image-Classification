#!/bin/bash

# activate test_env in Aanaconda
# export PATH=/home/dong/anaconda3/bin:/home/dong/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_jh

# 1 [END] large-batch training (64,32,16,128)
# 2 [END] cos-lr-decay 
# 3 [END] Transfer Learning
# 4 [ON AIR] Knowledge Distillation - Finding Teacher Model

# 5 [ON AIR] label-smoothing
############# test 1 #############
# experimental group
# python trainer_label_smoothing.py --model resnet50 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.1 --transfer_learning_scheme no
# python trainer_label_smoothing.py --model resnet34 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.1 --transfer_learning_scheme no
# # control group
CUDA_VISIBLE_DEVICES=1 python trainer_label_smoothing.py --model resnet34 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.0 --transfer_learning_scheme no
CUDA_VISIBLE_DEVICES=1 python trainer_label_smoothing.py --model resnet50 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.0 --transfer_learning_scheme no

# ############# test 2 #############
# # control group
# python trainer_label_smoothing2.py --model resnet34 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.0 --transfer_learning_scheme no --dropout yes
# CUDA_VISIBLE_DEVICES=3 python trainer_label_smoothing2.py --model resnet50 --batch_size 16 --lr 2.5e-4 --epochs 100 --scheduler no --pretrained no --label_smoothing_const 0.0 --transfer_learning_scheme no --dropout yes


# ############# test 3 #############
# # experimental group
# python trainer_label_smoothing.py --model resnet34 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet34 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet50 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet50 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# # control group
# python trainer_label_smoothing.py --model resnet34 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet34 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet50 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme
# python trainer_label_smoothing.py --model resnet50 --batch_size 128 --lr --epochs --scheduler --pretrained --label_smoothing_const --transfer_learning_scheme