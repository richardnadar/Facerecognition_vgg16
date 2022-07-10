# Face Recognition

## Task description 
-In this task I used the #VGG16 pre trained model and then using the concept of #transferlearning I recreated the model and used for predicting my own face.

- This task it took about 45 mins or so for training model with 8 epochs and then saving the model.

- After saving of the model, it was loaded for predicting the image/face.

- The model predicted the right image as the #VGG16 model is highly efficient model.

## Introduction
- VGG16 is a convolution neural net (CNN) architecture which was used to win ILSVR (Imagenet) competition in 2014. 
- It is considered to be one of the excellent vision model architecture till date. 
- Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. 
- It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FCLs (fully connected layers) followed by a softmax activation function for output. 
- The number 16 in VGG16 refers to the number of layers it has i.e 16 layers having weights. 
- This network is a pretty large network and it has about 138 million (approx) parameters.

Face Recognition program uses the concept of Transfer Learning. Transfer learning makes use of the knowledge gained while solving one problem and applying it to an another different but related problem.

## Requirement of transfer learning
- When we train the CNN network on a large dataset (like ImageNet), we train all the parameters of the neural network and that's how the model learns. 
- It may take hours of time and also consume alot of your GPU resources.
- So to reduce the time, we freeze all the layers of the pre-trained model (VGG16) and after that we add a new layer on top so that it can be trained with best weight and bias to detect the new object more accurately.

## Steps for creating the model
Step 1: I am using VGG16 as the pre-trained model and will do the further modifications on it to build the face recognition system.

Step 2: I created a dataset of my own face and of my friend and split it into 2 parts namely: training data and testing data. Once the model will be trained then I will use my test dataset to see whether the model is predicting accurately or not.

Step 3: First I will import all the necessary libraries which is needed to implement VGG16. I will be using Sequential technique as I am creating a sequential model. 
Sequential model means that all the layers of the model will be arranged in a sequence. 
Here I have imported ImageDataGenerator from keras.preprocessing. The objective of ImageDataGenerator is to import data with labels easily into the model. 
It is a very useful class as it has many function to rescale, rotate, zoom, flip etc. The most useful thing about this class is that it doesn’t affect the data stored on the disk. This class alters the data on the go while passing it to the model.

Step 4: Next I will be using Adam optimiser to reach to the global minima while training the model. 
If I am stuck in local minima while training then the adam optimiser will help us to get out of local minima and reach global minima. 
We will also specify the learning rate of the optimiser, in this case it is set at 0.001. 
If our training is bouncing a lot on epochs then we need to decrease the learning rate so that we can reach global minima.

Step 5: I am using model.fit_generator as I am using ImageDataGenerator to pass data to the model. I will pass train and test data to fit_generator. 
In fit_generator **"steps_per_epoch will"** set the batch size to pass training data to the model and **"validation_steps"** will do the same for test data. It can be tweaked based on system specifications.

Step 6: After executing the above code the model will start to train and you will start to see the training/validation accuracy and loss matrix.

Step 7: After model is comepltely trained, I saved it and loaded it for prediction. I passed the test data and the image predicted by the model is right.

