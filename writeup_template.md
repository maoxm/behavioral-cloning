#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image1.png "Model"
[image2]: ./image2.jpg "Center driving"
[image3]: ./image3.jpg "Recovery Image from left"
[image4]: ./image4.jpg "Recovery Image from right"
[image5]: ./image5.jpg "Normal image"
[image6]: ./image6.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 for video of self driving

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My final model is adapted from nvidia self driving model. It consists 5 convolution layers as: 24x5x5, 36x5x5, 48x5x5,
64x3x3, 64x3x3 with RELU layers (train.py function model3) . Then it is flatten to connect to 4 full connected layers as
100, 50, 10, 1

Data is preprocessed using normalization and cropping.


####2. Attempts to reduce overfitting in the model

To reduce overfitting, the model is trained on small number of epoch(usually 1 epoch). Also it is trained and validated
on different data sets by splitting the original data by 10%. (train.py 183-184 in function train() "else" branch). The
model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model uses adam optimizer to minimize mean squared error

####4. Appropriate training data

Training data is captured by driving on the road and correction from recovery(near the curb of the road)

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

1) Design a simple flow to test out the whole process including data collection, training, running model on simulator

2) Write simple neural network model as all full connected layer to prepare the code structure. Collect data and run
through training code to see the initial result.

3) Change model to be more complicated by adding convolutional layer and RELU layer. The car keeps driving to turn on
one direction.

4) Now I realized I need to pre-process the image and the data augment. So I normalized data and flip the image to
augment the data. Then I crop top and bottom part the image learned from the first project. The car runs better but only
holds around 10 sec.

5) I try to use the other 2 camera images for recovery. But it doesn't help a lot

6) I decide to dry more advanced model so I choose Nvidia's self driving model. 

7) By default I set the epoch to be 5 and I found the validation set accuracy is not decreasing a lot since the second
epoch. So to avoid over fitting, I tried epoch to be 1 and it works.


####2. Final Model Architecture

My final model is adapted from nvidia self driving model. It consists 5 convolution layers as: 24x5x5, 36x5x5, 48x5x5,
64x3x3, 64x3x3 with RELU layers (train.py function model3) . Then it is flatten to connect to 4 full connected layers as
100, 50, 10, 1


![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 1 lap of center driving(really good driving) around 4k images. Example:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides. Around 2k images:

![alt text][image3]
![alt text][image4]

The whole process took me around 2 hours.

To augment the data sat, I also flipped images. Example of image before and after flip.

![alt text][image5]
![alt text][image6]


After the collection process, I had around 6k images before the flip.


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under
fitting. The ideal number of epochs was 1 as evidenced by the validation accuracy is not improved much after epoch 2. I
used an adam optimizer so that manually training the learning rate wasn't necessary.
