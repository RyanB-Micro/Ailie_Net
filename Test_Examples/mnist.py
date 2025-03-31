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



epochs = 40
#epochs = 20
learn_rate = 0.01
image_data_file = 'C:/Users/Beds-/Desktop/mnist_train.csv'
image_test_file = 'C:/Users/Beds-/Desktop/mnist_train.csv'


training_labels = []
training_data = []
testing_labels = []
testing_data = []
# Create list to track errors over time
error_log = []

# get csv data
print("\nSTATUS: Reading Training Datafile")
train_dataset = pd.read_csv(image_data_file)
print("\nSTATUS: Reading Testing Datafile")
test_dataset = pd.read_csv(image_data_file)

train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)
print(f"\nAvailable Training Samples: {train_dataset.shape}")
print(f"Available Testing Samples: {test_dataset.shape}")

training_samples = train_dataset[0:1000]
testing_samples = test_dataset[0:1000]
print(f"\nTraining Data Size: {training_samples.shape}")
print(f"Testing Data Size: {testing_samples.shape}")


print("\nSTATUS: Building Dataset")
train_labels = []
for sample in training_samples:
    train_labels.append(sample[0])
    training_data.append(sample[1:])

test_labels = []
for sample in testing_samples:
    test_labels.append(sample[0])
    testing_data.append(sample[1:])


label_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# one hot encode training labels
training_labels = tu.simple_hot(train_labels, label_categories)

testing_labels = tu.simple_hot(test_labels, label_categories)


# test show first sample
# cross-checks that encodings and encodings work
print("\nSample Check")
first_label = train_labels[0]
print(f"Label: {first_label}")

first_encoded = tu.simple_hot([first_label], label_categories)
print(f"Label Encoding: {first_encoded}")

first_decoded = tu.hot_decode(first_encoded[0], label_categories)
print(f"Label: {first_decoded}")

#plt.imshow(training_data[0].reshape((28,28)), cmap='gray')
#plt.show()


# Creating the Neural Layers
input_size = 784 # amount of pixels in the images
output_size = 10 # ten characters categories to choose from
layer0 = Dense(512, input_size)
activ0 = Sigmoid_Layer()
layer1 = Dense(64, 512)
activ1 = Sigmoid_Layer()
layer2 = Dense(output_size, 64)
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
tu.train_network(neuralNet, training_data, training_labels, epochs, learn_rate, error_log, 'Cross')

# Prompting the uer to choose to whether plot is shown
user_prompt = "\n\tDisplay Plot for Error History? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    tu.plot_history(error_log, label_categories, "MNIST Training - Classification Error")


test_log = []
user_prompt = "\n\tTest Network against test data? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    test_log = tu.test_predictions(neuralNet, testing_data, testing_labels, test_log, 'Cross')

    tu.plot_history(test_log, label_categories, "MNIST Testing - Classification Error")

    test_results = {
        "Sample Label" : [],
        "Predicted Result" : []
    }
    for i in range(0, 10):
        test_data = testing_data[i]
        test_label = test_labels[i]
        print(f"\nTest Sample Label: {test_label}")
        prediction = neuralNet.forward(test_data)
        #print(f"Network Prediction: {prediction}")
        decoded_prediction = tu.hot_decode(prediction, label_categories)
        print(f"Predicted Label: {decoded_prediction}")
        test_results["Sample Label"].append(test_label)
        test_results["Predicted Result"].append(decoded_prediction)
    table = pd.DataFrame(test_results)
    print(table)
