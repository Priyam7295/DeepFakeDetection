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


   


## Clone the repository
- git clone https://github.com/Priyam7295/DeepFakeDetection.git 

We have used GAN images for training our model and then the test accuracy we are getting is around 89-90 percent . 
We further tested our modle on diffusion dataset to see whether our model is somehow generalising on diffusin datasets.

## We are getting around 90 percent accuracy 
![accvsepoch](https://github.com/Priyam7295/DeepFakeDetection/assets/136225328/a387280e-7d97-4dab-9bd2-11810a23ad9b)

***This can be increased by taking more variety of images and of different GAN and diffusion family***

