# Applying Ideas from Large Kernels Matters to Existing Architectures

Objective: This project investigates the trade-off between training time and accuracy when one reduces a kxk convolutional layer to 1xk convolutional layer followed by a kx1 convolutional layer in a U-Net CNN architecture applied to the image segmentation task on Kaggle Carvana Image Masking Challenge.

---
## Contributions

The project was done in collaboration with Erik Bertolino and Oscar Ameln for Deep Learning course at ETH Zurich. This project was based on [Heng CherKeng's code for PyTorch](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208) and [Petros Giannakopoulos's code for Keras](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge). 

---

## Introduction

This paper is inspired by concepts described in [Large Kernels Matters - Improve Image Segmentation by Global Convolution Network](https://arxiv.org/pdf/1703.02719.pdf). The authors of LKM argue that large kernels are beneficial for the accuracy, but have largely been discarded due to too long training time. The core concept is that a kxk convolutional layer can be approximated by a 1xk convolutional layer followed by a kx1 convolutional layer. The advantage of using this approximation is that the numbers of parameters are reduced from O(k^2) to O(2k). Using those two kernels reduces the number of parameters, which in turn should lead to shortened training time.

## Models

---

Three similar models were used based on U-Net architecture with convolutional layer with kernel sizes: 3x3, 5x5 and 1x5. 

![](https://raw.githubusercontent.com/mikpat/DL-Project/master/figures/DL.png)

3x3 model: the encoder and decoder consists of two sequences of 3x3 convolutions, batch normalizations and ReLu activations. In between U-blocks in the encoder 2x2 max pooling is used to reduce dimensions. Number of filters doubles with each level. On the other hand, deconvolution increases resolution of feature maps in the decoder part of U-Net architecture. Finally, identity operation adds feature maps from encoder to decoder in order to combine coarse and fine features. This is shown in the Figure below.

5x5 model:  all 3x3 convolutional layers are substitute by one 5x5 layer in each U-block.  

1x5 model: all 5x5 convolutional layers are substituted in the second model by a 1x5 convolutional layer followed by a 5x1 convolutional layer.

## Results

---


## Conclusions

---
