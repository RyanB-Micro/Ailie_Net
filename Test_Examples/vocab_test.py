
# Importing external library for graph capabilities
import matplotlib.pyplot as plt
# Importing external numpy maths library
import numpy as np

# Importing included Ailey_Net package
from Ailie_Net import *


""" USER TUNABLE PARAMETERS
:param epochs: The chosen number of training iterations to fit the Neural Net (Recommended: 100)
:param learn_rate: The size of the step size within each training iteration (Typically 0.01 -> 0.001)
"""
epochs = 100
learn_rate = 0.01

# Create list to track errors over time
error_log = []

# TRAIN SCRIPT
# hello
# what is your name
# hi
# who are you
# bye
# goodbye

vocab = ["hello", "what", "is", "your", "name", "hi", "who", "are", "you", "bye", "goodbye"]
vocab_one = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

phrases = ["hello", "my name is Ailie", "hey", "i am Ailie", "bye", "see ya"]
phrases_one = np.array([0, 0, 0, 0, 0, 0])



inputs = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],])

targets = np.array([[1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],])


layer0 = Dense(8, 11)
#layer1 = Dense(4, 8)
layer2 = Dense(6, 8)

neuralNet = AilieNet()
neuralNet.add(layer0)
#neuralNet.add(layer1)
neuralNet.add(layer2)

prediction = neuralNet.forward(inputs[0])
print("Prediction ", prediction)
#layer0.size()

error = squared_error(prediction, targets[0])
print("Error: ", error)
deriv_error = squared_error_prime(prediction, targets[0])
print("Deriv Error: ", deriv_error)


for itt in range(0, epochs):
    print("\nIteration: ", itt)
    epoch_error = 0
    for sample, target in zip(inputs, targets):
        prediction = neuralNet.forward(sample)
        print("Prediction: ", prediction)
        print("Target: ", target)
        error = squared_error(prediction, target)
        epoch_error += error
        deriv_error = squared_error_prime(prediction, target)
        back_error = neuralNet.backward(deriv_error, learn_rate)
        print("Error: ", error)

    error_log.append(epoch_error)

# Add the error scores to be plotted to the graph
plt.plot(error_log)
# Add a title to the graph
plt.title("Vocab Test - Training Error Over Time")
# Add labels to the axis of the graph
plt.xlabel("Number of Epochs")
plt.ylabel("Error Value")
# Add labels to the plotted lines in the top corner
plt.legend(phrases, loc="upper right")
# Display the graph
plt.show()