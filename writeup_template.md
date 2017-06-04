
# Behavioral cloning

  This project was very challenging as it shows the importance of data in neural network training.
 A simple network performs better than a complicated network here. Also, we need recovery data, to train our network to
 perform turns at sharp corners, as those are not present in udacity's data.I took some help from online resources like medium 
 as well as my mentor Fernando, to complete this project.


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### 1. Files Submitted & Code Quality

####  Submission contains all the files required for running the simulator in autonomus mode.

Project contains the following files.
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing


##### python drive.py model.h5

#### 3. Submitted Code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 4 convolutional filters  and then 3 fully connected layers given in the final_model(). I use Exponential Relu Units ('ELU') to introduce non linearity. 
You can see them as a parameter in the conv layers.('activation'='elu')


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).
The model was trained and validated on the data set , such that model was not overfitting. The epochs are also kept low, such that validation loss doesn't keep on increasing. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 
Also, the final model has layers of small size, thus fewer training parameters. This also reduces probability of overfitting.



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Near the sharp turns, I recorded some recovery data to keep the car follow the track.
To increase the data, I have used a simple data augmentation by flipping the image and negating the steering angle to double the data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially following  the video lectures , I tried two architectures : LeNet and Nvidia Net. I observed that strangely, using my preprocessing, the Nvidia-Net, performed a bit worse than the Nvidia-net on some turns. So I decided to stick with LeNet. However, LeNet is suitable for classification, not the regression kind of predictions.Hence, I replaced the Relu activation layers with Elu ones. So, I decided to use a modification of LeNet for my solution. 

Now, we know that LeNet works , when input is 32x32. So, First I cropped the top 80 and bottom 48 pixels of the images from dataset  to get a 32x320 image. Then I use tensorflow's resize_images() to resize them to 32x32.

Now, in the last project I faced the difficulty of choosing the right colourspace. So in the first layer, I chose a 1x1 conv. filter to pick me the right one. Then I used a 3x3 conv filter to learn the low level features. I keep the kernel small, so as to not lose low level features such as road edges etc.

In the later layers, to learn more high level features, I use filters of 5x5 kernel. Also similar to LeNet I add 3 fully connected layers.

Now, since this is not a classification problem, so I replaced max pooling with sub-sampling.
The provided dataset is only containing 8k images, so I added flipped images with flipped steering angles, for the flipped scenario.

Running the model through the simulator, gave a fair idea, of what to improve. Usually, the model failed near some sharp edges, after the bridge. So, some recovery laps needed to be recorded. Then retraining the model, finally gave a solution where the car could manouver properly.

For improvements, I could try some more data augmentation with better architectures.




#### 2. Final Model Architecture

* Layer 1 :  1x1 convolution filter with depth 3. We let the network figure out the best colour-space to transform the input into.

* Layer 2 : 3x3 convolution filter with depth 3 to learn the low level features. We choose a small kernel size to retain as much detail as we can.

* Layer 3 : 5x5 convolution filter with depth 6 to learn some higher level features.

* Layer 4 : 5x5 convolution filter with depth 16 to learn more higher level features.

* Layer 5 : Fully connected layer of 100.

* Layer 6 : Fully connected layer of 25.

* Layer 7 : Fully connected layer of 1.

The latter 3 dense layers have been inspired from the Lenet model. Instead of 'RELU' activation function, we choose and exponential 'RELU' i.e. 'ELU' .
This is needed as we do a regression kind of prediction of the output rather than classification.

[//]: # (Image References)

[image1]: ./cropped_img.png "Cropped image"
[image2]: ./resized_img..png "Resized image"
[image3]: ./flipped_img.png "Flipped image"
[image4]: ./examples/youtube-screenshot.png "YouTube Video Link"




![alt text][image1]

####3. Creation of the Training Set & Training Process

For the final model, I do some preprocessing and from each center image generate
two images.

1. First I crop the image as shown below. From a 160x320 image, I keep only 
a 32x320 image, by cropping the top 80 and bottom 48 pixels. This contain info
of sky/ground that are not relevant to our problem.

![Data Visualization][image1]
2. Now, since our network requires a 32x32 image, I use opencv's resize function
to resize images.

![Input Crop and Resize - Example 1][image2]

3. Now, to generate additional samples, I flip the above image and for it flip the 
steering too. You can see the flipped image as below. This provides us more data 
representing a different scenario. 

![Input Crop and Resize - Example 2][image3]

Then I train the model for 4-5 epochs. Since, I use an Adam optimizer, I don't need to tune the learning rates. The performance is evaluated on the simulator.
Now for the laps that the car goes, out, we collect recovery data and then retrain the
newtork. This process of refining our model, goes on till the car goes along the road
perfectly.

A video has been attached in the writeup and uploaded to youtube .
[![Video of successful lap around the track][image4]](https://youtu.be/4BfWHKVV-Hk "Successful Lap on the Test Track - Behavioral Cloning")
