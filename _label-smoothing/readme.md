# README

**train_LS.py** 을 통해, ResNet모델을 CosineLRdecay scheduler을 사용해 학습할 수 있다.

# How to Train

`` train_LS.py --model {resnet18, 34, ...} --batch_size {BS} --lr {LR} --epochs {epochs} --root_path {your_root}/datasets/archive/CUB_200_2011/ --pretrained {yes/no} --scheduler {yes/no} --label_smoothing_const 0.1 --transfer_learning_scheme {0,1,2,3} --dropout {yes/no} -seed {#seed} ``

## Some Arguments Description

- ``--transfer_learning_scheme``  
    - 0 : Train all layers  
    - 1 : Train FC layer only
    - 2 : Train FC layer + Few adjacent layers  
    - 3 : Discriminative Learning  

- ``--dropout``    
    Dropout used in convolutional filters according to below paper. (Ratio "0.1" is being used.)  
    "<i>Sungheon Park, Nojun Kwak. Analysis on the Dropout Effect in Convolutional Neural Networks. ACCV,
2016</i>"  

- ``--label_smoothing_const``  
    label smoothing constant. default "0.1"

