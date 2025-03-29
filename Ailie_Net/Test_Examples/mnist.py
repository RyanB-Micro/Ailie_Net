# import libraries
import pandas as pd
from matplotlib import pyplot as plt

# Importing external numpy maths library
import numpy as np

# Importing customs functions to support testing
import test_utils as tu

# Importing included Ailey_Net package
from Ailie_Net import *
from Ailie_Net.layer import Dense, Sigmoid_Layer, ReLU_Layer

epochs = 30
learn_rate = 0.01
image_data_file = 'C:/Users/*****/Desktop/mnist_train.csv'


training_labels = []
training_data = []
# Create list to track errors over time
error_log = []

# get csv data
print("\nSTATUS: Reading Datafile")
data = pd.read_csv(image_data_file)

data = np.array(data)
m, n = data.shape

training_samples = data[0:1200]


print("\nSTATUS: Building Dataset")
data_labels = []
for sample in training_samples:
    data_labels.append(sample[0])
    training_data.append(sample[1:])


label_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# one hot encode training labels
training_labels = tu.simple_hot(data_labels, label_categories)


# test show first sample
print("\nSample Check")
print(f"Label: {data_labels[0]}")
print(f"Label Encoding: {tu.hot_decode(data_labels[0], label_categories)}")
#plt.imshow(training_data[0].reshape((28,28)), cmap='gray')
#plt.show()


# Creating the Neural Layers
input_size = 784 # amount of pixels in the images
output_size = 10 # ten characters categories to choose from
layer0 = Dense(256, input_size)
activ0 = Sigmoid_Layer()
layer1 = Dense(128, 256)
activ1 = Sigmoid_Layer()
layer2 = Dense(output_size, 128)
activ2 = Sigmoid_Layer()

# Creating the Neural Network
neuralNet = AilieNet()
neuralNet.add(layer0)
neuralNet.add(activ0)
neuralNet.add(layer1)
neuralNet.add(activ1)
neuralNet.add(layer2)
neuralNet.add(activ2)


# Prompting the user to begin training
input("\n\tTHE COMPUTER IS READY TO TRAIN... (Press Return to Continue)\n")
tu.train_network(neuralNet, training_data, training_labels, epochs, learn_rate, error_log)

# Prompting the uer to choose to whether plot is shown
user_prompt = "\n\tDisplay Plot for Error History? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    tu.plot_history(error_log, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

