
# Importing external numpy maths library
import numpy as np

# Importing customs functions to support testing
import test_utils as tu

# Importing included Ailey_Net package
from Ailie_Net import *
from Ailie_Net.layer import Dense, Sigmoid_Layer, ReLU_Layer

""" USER TUNABLE PARAMETERS
epochs: The chosen number of training iterations to fit the Neural Net (Recommended: 100)
learn_rate: The size of the step size within each training iteration (Typically 0.01 -> 0.001)
chat_data_file: The file with prompt-response examples to train the model.
"""
epochs = 500
learn_rate = 0.01
chat_data_file = "chat_data.json"

# Create list to track errors over time
error_log = []

user_prompts, target_responses = tu.load_examples_json(chat_data_file)
print(f"\nuser prompts {user_prompts}")
print(f"\ntarget responses {target_responses}")

vocabulary = tu.build_vocab(user_prompts)
print(f"\nvocabulary {vocabulary}")

phrasebook = tu.build_phrasebook(target_responses)
print(f"\nphrasebook {phrasebook}")

training_data, training_targets = tu.gen_chat_training(user_prompts, target_responses, vocabulary, phrasebook)
print(f"\ntraining data {training_data}")
print(f"\ntraining targets {training_targets}")

# Creating the Neural Layers
input_size = len(training_data[0])
output_size = len(training_targets[0])
layer0 = Dense(16, input_size)
activ0 = ReLU_Layer()
layer1 = Dense(16, 16)
activ1 = ReLU_Layer()
layer2 = Dense(output_size, 16)
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
tu.train_network(neuralNet, training_data, training_targets, epochs, learn_rate, error_log)

# Prompting the uer to choose to whether plot is shown
user_prompt = "\n\tDisplay Plot for Error History? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    tu.plot_history(error_log, phrasebook)

# Prompting the user to begin chatting
input("\n\tTHE COMPUTER IS READY TO CHAT... (Press Return to Continue)\n")

# Creating a loop cycle where the computer prompts the user for input, and generates a response.
chatting = True
while chatting:
    # Getting a message from the user
    input_phrase = input("\nChat to your computer >... ")

    # Exiting the chat loop if requested
    if input_phrase == "Stop":
        chatting = False

    # Encoding the users message
    vocab_encoding = tu.input_encoder(input_phrase.lower(), vocabulary)
    print(f"\nYou > {input_phrase}\nInput Encoding: {vocab_encoding}")

    # Generating the computers response
    raw_response = neuralNet.forward(vocab_encoding)
    response = tu.output_decoder(raw_response, phrasebook)
    print(f"\nComputer > {response}\nOutput Encoding: {raw_response}")



