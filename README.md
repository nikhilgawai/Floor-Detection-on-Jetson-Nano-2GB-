# Floor Detection on Jetson Nano 2GB Developer Kit using Yolov5.

#### Floor detection system which will detect the quality of Floor, identify whether
#### its clean or unclean and then recommend the steps to take after identifying the property of the Floor.

## Aim and Objectives

### Aim

To create a Floor detection system which will detect the quality of Floor, identify whether
its clean or unclean and then recommend the steps to take after identifying the property of the
Floor.

### Objectives

• The main objective of the project is to create a program which can be either run on
Jetson nano or any pc with YOLOv5 installed and start detecting using the camera
module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine
learning.

• To show on the optical viewfinder of the camera module whether a Floor is clean or
unclean.
## Abstract

• A Floor’s cleanliness can be detected by the live feed derived from the system’s
camera.

• We have completed this project on jetson nano which is a very small computational
device.

• A lot of research is being conducted in the field of Computer Vision and Machine
Learning (ML), where machines are trained to identify various objects from one
another. Machine Learning provides various techniques through which various objects
can be detected.

• One such technique is to use YOLOv5 with Roboflow model , which generates a small
size trained model and makes ML integration easier.

• Clean and Beautiful looking floors provide a refreshing and mesmerizing look and
helps in creating a good ambience in a given environment.

• Clean floors provides traction improving the safety of the place by eliminating slipping
and also removes allergens to further provide with a fresher and healthier environment.
## Introduction


• This project is based on a Floor detection model with modifications. We are going to
implement this project with Machine Learning and this project can be even run on jetson
nano which we have done.

• This project can also be used to gather information about Floor condition, i.e., Clean,
Unclean.

• Floor can be classified into clean, unclean, clear, dirty, spotless etc based on the image
annotation we give in roboflow.

• Floor detection in our model sometimes becomes difficult because of various textures in
floor like spots texture, lines texture, or various other graphical textures. However, training
our model with the images of these textured floor makes the model more accurate.

• Neural networks and machine learning have been used for these tasks and have obtained
good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Floor detection as well.
## Literature Review

• This project is based on a Floor detection model with modifications. We are going to
implement this project with Machine Learning and this project can be even run on jetson
nano which we have done.

• This project can also be used to gather information about Floor condition, i.e., Clean,
Unclean.

• Floor can be classified into clean, unclean, clear, dirty, spotless etc based on the image
annotation we give in roboflow.

• Floor detection in our model sometimes becomes difficult because of various textures in
floor like spots texture, lines texture, or various other graphical textures. However, training
our model with the images of these textured floor makes the model more accurate.

• Neural networks and machine learning have been used for these tasks and have obtained
good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Floor detection as well.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.

## Jetson Nano 2GB

![IMG_20220125_115056](https://user-images.githubusercontent.com/89011801/151312523-33b6cd88-9b92-453c-bb8e-b6e5a1fb782b.jpg)

## Proposed System

   1. Study basics of machine learning and image recognition.
    
2. Start with implementation
        
        ➢ Front-end development
        ➢ Back-end development
3. Testing, analysing and improvising the model. An application using python and
Roboflow and its machine learning libraries will be using machine learning to identify
the cleanliness of Floor.

4. Use datasets to interpret the Floor and suggest whether the Floor are clean or unclean.
## Methodology

The floor detection system is a program that focuses on implementing real time floor
detection.

It is a prototype of a new product that comprises of the main module:

Floor detection and then showing on viewfinder whether clean or unclean.

Floor Detection Module

#### This Module is divided into two parts:-


#### 1] Floor detection

• Ability to detect the location of floor in any input image or frame. The output is
the bounding box coordinates on the detected floor.

• For this task, initially the Dataset library Kaggle was considered. But integrating
it was a complex task so then we just downloaded the images from
gettyimages.ae and google images and made our own dataset.

• This Datasets identifies floor in a Bitmap graphic object and returns the bounding
box image with annotation of floor present in a given image.

#### 2] Cleanliness Detection

• Classification of the floor based on whether it is clean or unclean.

• Hence YOLOv5 which is a model library from roboflow for image classification
and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use
in production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments
is straightforward.

• YOLOv5 was used to train and test our model for various classes like clean,
unclean. We trained it for 149 epochs and achieved an accuracy of
approximately 93%.

## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```

```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```

## Floor Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder
link of project

Insert gif or link to demo


## Running Floor Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```


## Demo



https://user-images.githubusercontent.com/89011801/151313206-415fea54-3c51-4a7c-b606-d0ac64d8d009.mp4


 


## Advantages

➢ The Floor detection system will be of great advantage where a user has lack of time,
motivation, unwell or differently abled.

➢ It will be useful to users who are very busy because of work or are because of prior
schedules.

➢ Just place the viewfinder showing the Floor on screen and it will detect it.

➢ It will be faster to just then clean floor using minimal or very less workforce.
## Application

➢ Detects Floor clarity in a given image frame or viewfinder using a camera module.

➢ Can be used to clean Floor when used with proper hardware like machines which can
clean.

➢ Can be used as a reference for other ai models based on floor detection
## Future Scope


➢ As we know technology is marching towards automation, so this project is one of the step
towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

➢ Cleaning floors inside vehicles, trains, buses automatically as well as outer surfaces of
ships and submarines can be considered a good use of our model.
## Conclusion

➢ In this project our model is trying to detect floors for whether they are clean or unclean
and then showing it on viewfinder live as what the state of floor is.

➢ This model solves the basic need of having a clean and clear floor for our users who
because of lack of time or other reasons are unable to keep their floor clean.

➢ It can even ease the work of people who are in the sanitization industry or the cleaning
industry and save them a lot of time and money.
## Refrences

#### 1]Roboflow :- https://roboflow.com/

#### 2] Datasets or images used: https://www.gettyimages.ae/search/2/image?family=creative&phrase=floor

#### 3] Google images
## Articles

## Articles :-

#### [1] https://www.therichest.com/lifestyles/10-reasons-to-focus-on-the-importance-of-proper-floor-maintenance/

#### [2]https://shinycarpetcleaning.com/benefits-of-having-floor-cleaning-maintenance/
