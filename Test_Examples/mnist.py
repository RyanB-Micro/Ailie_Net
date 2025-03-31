# import external libraries
import pandas as pd # Data management
from matplotlib import pyplot as plt # Data visualisation
import numpy as np # Fast an efficient maths library

# Importing customs functions specific to supporting test projects
import test_utils as tu

# Importing necessary Ailey_Net packages
from Ailie_Net.network import AilieNet
from Ailie_Net.utils import user_choice
from Ailie_Net.layer import Dense, Sigmoid_Layer


""" USER TUNABLE PARAMETERS
epochs: The chosen number of training iterations to fit the Neural Net (Recommended: 100).
learn_rate: The size of the step size within each training iteration (Typically 0.01 -> 0.001).
image_data_file: The file containing the labeled image data for training purposes.
image_test_file: The file containing the labeled image data for testing purposes.
"""
epochs = 40
learn_rate = 0.01
image_data_file = 'C:/Users/Beds-/Desktop/mnist_train.csv'
image_test_file = 'C:/Users/Beds-/Desktop/mnist_train.csv'

# Empty lists to for later storing separate training data and labels
training_labels = []
training_data = []
# Empty lists to for later storing separate training data and labels
testing_labels = []
testing_data = []

# A list used to track error over training iterations
error_log = []
# A list used to track error over testing iterations
test_log = []

# A dictionary to store the tabled results of testing the network
test_results = {
    "Sample Label": [],
    "Predicted Result": []
}

# A list containing the labels associated with the image classifications
label_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


""" LOADING DATA """
#-------------------

# Load training and testing csv data from the testing and training files
print("\nSTATUS: Reading Training Datafile")
train_dataset = pd.read_csv(image_data_file)
print("\nSTATUS: Reading Testing Datafile")
test_dataset = pd.read_csv(image_data_file)

# Convert the data into a numpy array for increased speed and functionality
train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)

# Display the total lengths of each dataset
print(f"\nAvailable Training Samples: {train_dataset.shape}")
print(f"Available Testing Samples: {test_dataset.shape}")

# Portion the datasets into smaller subsets
# This will trade overall accuracy with less overhead for faster training speeds
training_samples = train_dataset[0:1000]
testing_samples = test_dataset[0:1000]

# Display the sizes of the portioned datasets
print(f"\nTraining Data Size: {training_samples.shape}")
print(f"Testing Data Size: {testing_samples.shape}")


""" Building Datasets """
#------------------------

# Separate the training data and labels into Separate lists
print("\nSTATUS: Building Training Dataset")
train_labels = []
for sample in training_samples:
    train_labels.append(sample[0])
    training_data.append(sample[1:])

# Separate the training data and labels into Separate lists
print("\nSTATUS: Building Testing Dataset")
test_labels = []
for sample in testing_samples:
    test_labels.append(sample[0])
    testing_data.append(sample[1:])

# One-Hot encoding the dataset labels so they can be compared to the networks outputs
print("\nSTATUS: Encoding Dataset Labels")
training_labels = tu.simple_hot(train_labels, label_categories)
testing_labels = tu.simple_hot(test_labels, label_categories)


""" Datasets Checking """
#------------------------

# Test show the first training sample
# This cross-checks that encodings and encodings work and that the dataset is compatible
print("\nSample Check - Dataset Compatibility Validation")
print("Warning: If the following images and labels do not match, there may be a dataset issue.")

# Acquire the label first training sample as a control
first_label = train_labels[0]
print(f"Label Original: {first_label}")

# Encode the retrieved label, into its one-hot encoding form
first_encoded = tu.simple_hot([first_label], label_categories)
print(f"Label Encoded: {first_encoded}")

# Decode the encoded label into its original form to check its matches the control
# If they are not matched, it indicated a compatibility issue between the dataset and the encoding method
first_decoded = tu.hot_decode(first_encoded[0], label_categories)
print(f"Label Decoded: {first_decoded}")

# Automatically check for encoding-decoding errors
if first_label != first_decoded:
    print("Encoding Error Detected - Possible Dataset Incompatibility.")

# Display the sample to verify a visual match to the acquired label
# If there is a non match, then there is a compatibility issue between the dataset formating and the data loading.
print("\nVisual Match Test - The displayed image should visually match the label.")
print("(Close Plot Window to Continue)")
plt.imshow(training_data[0].reshape((28,28)), cmap='gray')
plt.show()


""" Building the Model """
#-------------------------

# Defining the intended input size and output sizes of the network
input_size = 784 # amount of pixels in the images
output_size = 10 # ten characters categories to choose from

print("\nBuilding Neural Network Layers")
# Creating the individual Neural and activation Layers
layer0 = Dense(512, input_size)
activ0 = Sigmoid_Layer()
layer1 = Dense(64, 512)
activ1 = Sigmoid_Layer()
layer2 = Dense(output_size, 64)
activ2 = Sigmoid_Layer()

print("\nBuilding Neural Network Architecture")
# Creating the Neural Network by combining the layers
neuralNet = AilieNet()
neuralNet.add(layer0)
neuralNet.add(activ0)
neuralNet.add(layer1)
neuralNet.add(activ1)
neuralNet.add(layer2)
neuralNet.add(activ2)


""" Training the Model """
#-------------------------

# Prompting the user to begin training
input("\n\tTHE COMPUTER IS READY TO TRAIN... (Press Return to Continue)\n")
# Uses a custom function to train the network by providing data, targets, and training parameters
tu.train_network(neuralNet, training_data, training_labels, epochs, learn_rate, error_log, 'Cross')

# Prompting the uer to choose to whether the training plot is shown
user_prompt = "\n\tDisplay Plot for Error History? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    # Display the visualised error over time calculated over training
    tu.plot_history(error_log, label_categories, "MNIST Training - Classification Error")


""" Training the Model """
#-------------------------
# Prompting the uer to choose to whether the network is tested against the testing dataset
user_prompt = "\n\tTest Network against test data? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    # Make predictions of the classifications of the testing data
    test_log = tu.test_predictions(neuralNet, testing_data, testing_labels, test_log, 'Cross')

    # Plot the error of the known test dataset labels against the predictions
    tu.plot_history(test_log, label_categories, "MNIST Testing - Classification Error")

    # Loop through a sample of side by side comparisons of known labels and network predictions
    print("\nComparisons Table - Dataset Labels vs Predicted Classifications")
    for i in range(0, 10):
        # Acquire test sample data and label
        test_data = testing_data[i]
        test_label = test_labels[i]
        print(f"\nTest Sample Label: {test_label}")

        # Generate a predicted label from the data
        prediction = neuralNet.forward(test_data)

        # Decode and display the prediction into a human-readable format
        decoded_prediction = tu.hot_decode(prediction, label_categories)
        print(f"Predicted Label: {decoded_prediction}")

        # Store the results into the dictionary
        test_results["Sample Label"].append(test_label)
        test_results["Predicted Result"].append(decoded_prediction)

    # Generate a Pandas dataframe to store the results into a table format
    table = pd.DataFrame(test_results)
    # Display the results in an organised table format
    print(table)
