# README

**making_cosLRdecay.ipynb** 는 cosine learning rate decay with linear warmup을 구현하는 과정이 있는 주피터노트북.  
**trainer_cosLRdecay.py** 을 통해, ResNet모델을 CosineLRdecay scheduler을 사용해 학습할 수 있다.

# How to Train

``` trainer_cosLRdecay.py --model {resnet18, 34, ...} --batch_size {BS} --lr {LR} --epochs {epochs} --root_path {your_root}/datasets/archive/CUB_200_2011/ --pretrained {yes/no} --cos_scheduler yes --lin_end {which #step linear warmup for COSscheduler ends}```