# Applying Ideas from Large Kernels Matters to Existing Architectures

Objective: This project investigates the trade-off between training time and accuracy when one reduces a kxk convolutional layer to 1xk convolutional layer followed by a kx1 convolutional layer in a U-Net CNN architecture applied to the image segmentation task on Kaggle Carvana Image Masking Challenge.

## Contributions

The project was done in collaboration with Erik Bertolino and Oscar Ameln for Deep Learning course at ETH Zurich. This project was based on [Heng CherKeng's code for PyTorch](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208) and [Petros Giannakopoulos's code for Keras](https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge). 


## Introduction

This paper is inspired by concepts described in [Large Kernels Matters - Improve Image Segmentation by Global Convolution Network](https://arxiv.org/pdf/1703.02719.pdf). The authors of LKM argue that large kernels are beneficial for the accuracy, but have largely been discarded due to too long training time. The core concept is that a kxk convolutional layer can be approximated by a 1xk convolutional layer followed by a kx1 convolutional layer. The advantage of using this approximation is that the numbers of parameters are reduced from O(k^2) to O(k). Using those two kernels reduces the number of parameters, which in turn should lead to shortened training time.

## Models


Three similar models were used based on U-Net architecture with convolutional layer with kernel sizes: 3x3, 5x5 and 1x5. 

![](https://raw.githubusercontent.com/mikpat/DL-Project/master/figures/DL.png)

3x3 model: each U-block in the encoder and decoder consists of two sequences of 3x3 convolutions, batch normalizations and ReLu activations. In between U-blocks in the encoder 2x2 max pooling is used to reduce dimensions. Number of filters doubles with each level. On the other hand, deconvolution increases resolution of feature maps in the decoder part of the U-Net architecture. Finally, identity operation adds feature maps from encoder to decoder in order to combine coarse and fine features. This is shown in the Fig.1.

5x5 model:  all 3x3 convolutional layers are substitute by one 5x5 layer in each U-block.  

1x5 model: all 5x5 convolutional layers are substituted in the second model by a 1x5 convolutional layer followed by a 5x1 convolutional layer.

## Results

As shown in the Fig.2, the 3x3 model seems to be more stable than the two others. The 3x3 model exhibits a steady increase in the Sorensen-Dice coefficient in the first run and experiences fewer spikes than 1x5 and 5x5 in the second round of training. Ideally, more training should be done to investigate whether these patterns are replicable. Unfortunately, more training was outside time scope for this project.

![](https://raw.githubusercontent.com/mikpat/DL-Project/master/figures/DL_results.png)

By inspecting Fig.2A, it can be seen that at around 75 000 batches dice coefficient is similar for all models. The advantage of the 1x5 model is that it is able to process more batches during 110 hours. Unfortunately, the behaviour of the validation dice coefficient after 110 hours is unknown and therefore it cannot be concluded that the 1x5 model is superior to 3x3 and 5x5 models for longer periods of training.

Moreover, it is shown in the Fig.2A that the 1x5 model can process around 140 000 batches during 110 hours of training, compared to the 5x5 model that reaches 70 000 batches. The reduction of parameters from O(k^2) to O(k) makes it possible to process 1.8 times more batches, given constrained training time. As k=5 is a moderately low k, an even better improvement in training is expected by applying the kernel separation on larger kernels, given that the number of parameters are dominating the training time asymptotically.

Another advantage of kernel separation is usage of smaller memory resources. For each 5x5 convolution layer, two layers of 3x3 convolutions were used. Since the number of filters in each layer is the same, the 5x5 model had 1389 more weights than 3x3 model. After the kernel separation, the ratio decreased to 0.55. This is evident when comparing sizes of files with model weights. 5x5 model used 43.8 MB, 3x3 used 30.9 MB and the model with kernel separation used 17.2 MB.

## Conclusions

The results suggest that the kernel separation can shorten training time without sacrificing accuracy. Usage of the kernel separation is especially beneficial in a situation of small computational resources or in a need of fast training time. Assuming that the number of parameters of the kernels dominates the constant operations asymptotically, even more favourable results are expected as k increases.
