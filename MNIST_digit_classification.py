from calendar import EPOCH
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix


# Loading MNIST data from keras .dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Shape of the numpy arrays 
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

#Training = 60000 Images & Testing = 10000 Images. the images are grayscale and size is 28 x 28

#print one of the dataset image 
# print(X_train[10])
# plt.imshow(X_train[10])
# plt.show()
#printing the label of the image as well
# print(Y_train[10])
# To display all unique numbers in the mnist dataset
# print(np.unique(Y_train))

# Inorder to reduce load over machine in order to process the images
# which are having color in range 0-255, we reduce it into range of 0-1
X_train = X_train/255
X_test = X_test/255

# print(X_train[10])


## Building the Neural Network
# Flatten - we cant't feed data as matrix so we convert it into single line using it.
# The input shape is the size of the image

# Dense - all the layers comes into previous or next layer, (no of neurons, activation function = relu)

# relu = rectified linear unit
# sigmoid

# 2 layers are used
# The final layer is output layer so that the neurons of the previous layers are connected to this layer
# The final layer should have total 10 neurons as we have a total of (0-9) 10 values

model = tf.keras.Sequential([keras.layers.Flatten(input_shape=(28,28,1)), 
                                                  keras.layers.Dense(50, activation = 'relu'), 
                                                  keras.layers.Dense(50, activation = 'relu'), 
                                                  keras.layers.Dense(10, activation = 'sigmoid')])

# Compliing the model

# optimizer is used to determine the most optimium model parameter

# ex. linear regression w b

# loss is used for label removing

# metrics = 
# accuracy = no of correct prediction/ total number of data passed

model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Train neural network
model.fit(X_train, Y_train, epochs=10)