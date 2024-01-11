# DeepFake Detection 
## Using Neural Network
Fake Image Buster : Detecting Deception with Machine Learning
Welcome to Fake Image Buster , a project dedicated to combating the growing threat of synthetic media.
In today's digital age , the images can easily be manipulated using many GANS or Diffusion models, while these can be useful in many ways but there is a huge concern for safety and this fake images can be misued in many ways .
This project leverages the power of simple Machine Learning to build a **binary classification** model that can distinguish between real and fake images with impressive accuracy.

## Project Highlights:

   1. ***Model***:   Using a pre-trained ResNet architecture and adding custom layers for image classification task. We took GAN images and passed it over resnet pre trained model with some more layer layered on top of it , and then we tested our model on different **DIFFUSION MODEL** as well.
   1. ***Data Augmentation***:   Enhances training data diversity by applying random rotations ,  
     horizontal rotations that improves generalizability.
   1. ***Performance***:   Delivers good accuracy , providing a reliable tool for image verification.
   1. ***Testing on DM***:  Testing on different diffusion models .


   


## Clone the repository
- git clone https://github.com/Priyam7295/DeepFakeDetection.git 

We have used GAN images for training our model and then the test accuracy we are getting is around 80-85 percent for GAN images. 
We further tested our modle on diffusion dataset to see whether our model is somehow generalising on diffusin datasets.
However our model nearly give chance probabilty (50 percent ) while testing for DM images because CANN focuses on Low level cues left by GAN as it is easy to learn and label all other class as real , hence accuracy is low . 
We will further use KNN based classification for creating model than can be used for detecting any type of GANS and DM images. 

## We are getting around 80 percent accuracy 
   ![accvsepoch](https://github.com/Priyam7295/DeepFakeDetection/assets/136225328/a387280e-7d97-4dab-9bd2-11810a23ad9b)

***This can be increased by taking more variety of images and of different GAN and diffusion family***

