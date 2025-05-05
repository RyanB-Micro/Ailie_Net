
"""
This program tests a single layer implementation of an Ailey_Net Neural Network.

Author: Ryan Brown
Date Created: 17.03.2025
------------------------
Last Edited: 20.03.2025
Last Change: Modification of the program brief
Repo: https://github.com/CMOSSE101/Ailie_Net

Concept: Single Layer Pattern Detector

Brief: This project detects patterns for whether an "object"  is detected on the left or right side
of an input. This is implemented using a networks comprised of a single Dense layer.

Problem Scenario: A "robot" is using a very simple sensor array to navigate a space.
In order to avoid obstacles, the robot needs an AI system capable of detecting objects,
so it can decide on which direction it needs to move.

 [Object]   [Object]   [No Object]

            o  o  o
             \ | /
            -------
           |(robot)\
           ---------

    (Decision: Objects detected on Left Side)

Using the 3 onboard sensors, the robot needs to be able to detect if there is any objects
facing its front left side, or its front right side. Once the AI has determined if there are any objects
on its left or right side, it can then make its decision on the best direction it should move.


Example: The pattern "[1, 0, 0]" represents an object detected on the left side of the robot.
This should result with the response "[1, 0]" signifying a detection on the "left" detector neuron.
The pattern "[0, 0, 1]" represents an object present on the right side.
Pattern "[0, 0, 0]" represents no object present to detect.

"""

# Importing external library for graph capabilities
import matplotlib.pyplot as plt
# Importing external numpy maths library
import numpy as np

# Importing included Ailey_Net package
from Ailie_Net import *

""" USER TUNABLE PARAMETERS
:param epochs: The chosen number of training iterations to fit the Neural Net (Recommended: 50)
:param learn_rate: The size of the step size within each training iteration (Typically 0.01 -> 0.001)
"""
epochs = 50
learn_rate = 0.01

# Create list to track errors over time
error_log = []

# Training data sample patterns
training_data = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
# Training data associated target classification
training_targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

# Creating a layer of 2 neurons, each with 3 inputs
layer = Dense(2, 3)

# Creating instance of a new Ailie_Net Neural Network
neuralNet = AilieNet()
# Add the created layer to the network
neuralNet.add(layer)

# Loop through chosen training iteration quantity
for e in range(0, epochs):
    # Display the current epoch counter
    print(f"\nEpoch: {e}/{epochs}")

    # Clear total epoch error variable
    epoch_error = 0

    # Sequence through each pattern per epoch
    for pattern, target in zip(training_data, training_targets):
        # Create a prediction for the current pattern
        prediction = neuralNet.forward(pattern)
        print(f"Prediction for pattern {pattern}: {prediction}")

        # Calculate error for prediction
        error = squared_error(prediction, target)
        # Sum errors across epoch
        epoch_error += error
        print(f"Expected Result: {target}, Calculated Error: {error}")

        # Calculate the derivative for the error
        deriv_error = squared_error_prime(prediction, target)
        # Feed the error backwards through the network
        # This will automatically adjust parameters within the layer
        neuralNet.backward(deriv_error, learn_rate)

    # Add the total error of the epoch to the error log
    error_log.append(epoch_error)


# Add the error scores to be plotted to the graph
plt.plot(error_log)
# Add a title to the graph
plt.title("Pillar Detector - Training Error Score Over Time")
# Add labels to the axis of the graph
plt.xlabel("Number of Epochs")
plt.ylabel("Error Value")
# Add labels to the plotted lines in the top corner
plt.legend(["Left Detector", "Right Detector"], loc="upper right")
# Display the graph
plt.show()
