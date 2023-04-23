# Bag of Tricks for Image Classification

:robot: My Implementation of the approach described in below paper. Hope you got helped!
> He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.  
> https://arxiv.org/abs/1812.01187  

## :dizzy:Update
 23.04.23: Initial upload !

## :dizzy:Introduction
Experiments for below methods introduced in this paper.
1. Large Batch Training
2. Cosine Learning Rate Decay
3. Label Smoothing
4. Transfer Learning

You can experiment upper methods using this repository.  


## :dizzy:Tested Dependencies

``GPU`` NVIDIA RTX-3060-12GB, A6000-48GB, Colab-K80  
``CUDA`` 11.2  
``pytorch`` 1.9.1+cu111  
``torchvision`` 0.10.1+cu111  
``python`` 3.9.1  
``OS`` Ubuntu 20.04, Ubuntu 18.04  
``Datasets`` CUB200-2011, ImageNet-1K

(As I don't use special libraries, errors may not occur in other environments.)

---

## :dizzy:Dataset setup

Plese download the dataset from https://www.kaggle.com/datasets/coolerextreme/cub-200-2011. And split your data into train, test folders.
```
${YOUR_ROOT}/
|-- workspace
|   |-- Bag-of-Tricks-for-Image-Classification  # current repository.
|-- dataset
|   |-- archive
|   |   |-- CUB_200_2011
|   |   |   |-- CUB_200_2011
|   |   |   |   |-- train
|   |   |   |   |-- test
|   |   |   |   |-- ...
```

## :dizzy:How to train the model by each Methods
Please visit each directory's ``readme.md``!

## :dizzy:Licence
This project is licensed under the terms of the MIT license.
