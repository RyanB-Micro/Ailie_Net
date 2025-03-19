
# Importing external library for graph capabilities
import matplotlib.pyplot as plt
# Importing external numpy maths library
import numpy as np
# importing the library to read JSON files
import json

# Importing included Ailey_Net package
from Ailie_Net import *


""" USER TUNABLE PARAMETERS
epochs: The chosen number of training iterations to fit the Neural Net (Recommended: 100)
learn_rate: The size of the step size within each training iteration (Typically 0.01 -> 0.001)
"""
epochs = 400
learn_rate = 0.01

# Create list to track errors over time
error_log = []

# Known words the user might say
# vocab = ["hello", "what", "is", "your", "name", "hi", "who", "are", "you", "bye", "goodbye"]
#
# Available responses for the computer
# phrasebook = ["hello", "my name is Ailie", "hey", "i am Ailie", "bye", "see ya"]

# Training data of sample phrases for the computer to detect
# training_data = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],])

# Training targets of desired outputs or detectable phrases
# training_targets = np.array([[1, 0, 1, 0, 0, 0],
#                             [0, 1, 0, 1, 0, 0],
#                             [1, 0, 1, 0, 0, 0],
#                             [0, 1, 0, 1, 0, 0],
#                             [0, 0, 0, 0, 1, 1],
#                             [0, 0, 0, 0, 1, 1]])




def load_examples_json(json_file: str):
    # Opens the file containing the chat script
    file = open(json_file)

    # generate dictionary from json file
    chat_script = json.load(file)

    prompt_list = []
    response_list = []

    for example in chat_script["Chat_Scripts"]:
        # Put stored phrases in respective lists
        prompt_list.append(example["input"].lower())
        response_list.append(example["response"])

    file.close()

    return prompt_list, response_list


def gen_chat_training(prompt_list, response_list, vocabulary, phrasebook):
    training_data = []
    training_targets = []

    for prompt, response in zip(prompt_list, response_list):
        data_temp = [0 for i in range(len(vocabulary))]
        targets_temp = [0 for i in range(len(phrasebook))]
        # find which indexes the prompts words exist in vocabulary
        for word in prompt.split():
            vocab_index = vocabulary.index(word)
            # one-hot encode detected vocab word
            data_temp[vocab_index] = 1

        # find which index the phrase exists in phrasebook
        phrase_index = phrasebook.index(response)
        # one-hot encode detected phrase
        targets_temp[phrase_index] = 1

        # add encodings to training lists
        training_data.append(data_temp)
        training_targets.append(targets_temp)

    return training_data, training_targets


def build_vocab(prompt_list: list) -> list:
    vocab_builder = []
    # collate prompt words into a list of unique words
    for prompt in prompt_list:
        for word in prompt.split():
            if word not in vocab_builder:
                vocab_builder.append(word)

    return vocab_builder


def build_phrasebook(response_list: list) -> list:
    phrasebook_builder = []
    # collate prompt words into a list of unique words
    for phrase in response_list:
        if phrase not in phrasebook_builder:
            phrasebook_builder.append(phrase)

    return phrasebook_builder

chat_data_file = "chat_data.json"
user_prompts, target_responses = load_examples_json(chat_data_file)
print(f"\nuser prompts {user_prompts}")
print(f"\ntarget responses {target_responses}")
vocabulary = build_vocab(user_prompts)
print(f"\nvocabulary {vocabulary}")
phrasebook = build_phrasebook(target_responses)
print(f"\nphrasebook {phrasebook}")
training_data, training_targets = gen_chat_training(user_prompts, target_responses, vocabulary, phrasebook)
print(f"\ntraining data {training_data}")
print(f"\ntraining targets {training_targets}")

def train_network(network: AilieNet, inputs, targets, error_log):
    for itt in range(0, epochs):
        print("\nIteration: ", itt)
        epoch_error = 0
        for sample, target in zip(inputs, targets):
            prediction = network.forward(sample)
            print("Prediction: ", prediction)
            print("Target: ", target)
            error = cost(prediction, target)
            epoch_error += error
            deriv_error = cost_prime(prediction, target)
            back_error = network.backward(deriv_error, learn_rate)
            print("Error: ", error)

        error_log.append(epoch_error)

def plot_history(error_log, ledgend_categories):
    # Display instructions
    print("\n\tTHE PLOT IS BEING DISPLAYED... (Close Plot Window to Continue)")
    # Add the error scores to be plotted to the graph
    plt.plot(error_log)
    # Add a title to the graph
    plt.title("Vocab Test - Training Error Over Time")
    # Add labels to the axis of the graph
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error Value")
    # Add labels to the plotted lines in the top corner
    plt.legend(ledgend_categories, loc="upper right")
    # Display the graph
    plt.show()


def input_encoder(input_phrase, vocab):
    split_text = input_phrase.split()

    vocab_encoding = np.zeros(len(vocab))

    for user_word in split_text:
        for index, stored_phrase in enumerate(vocab, start=0):
            if user_word == stored_phrase:
                vocab_encoding[index] = 1

    return vocab_encoding

def output_decoder(phrasebook_encoding, phrasebook):
    # Find the strongest output in the neural netoworks output
    chosen_phrase = np.argmax(phrasebook_encoding)

    # Double check the output value is above a strength threshold
    if phrasebook_encoding[chosen_phrase] > 0.7:
        return phrasebook[chosen_phrase]
    else: # No single output was strong enough
        return "Sorry, i do not understand."


# Creating the Neural Layers
input_size = len(training_data[0])
output_size = len(training_targets[0])
layer0 = Dense(8, input_size)
layer1 = Dense(8, 8)
layer2 = Dense(output_size, 8)

# Creating the Neural Network
neuralNet = AilieNet()
neuralNet.add(layer0)
neuralNet.add(layer1)
neuralNet.add(layer2)

# Prompting the user to begin training
input("\n\tTHE COMPUTER IS READY TO TRAIN... (Press Return to Continue)\n")
train_network(neuralNet, training_data, training_targets, error_log)

# Prompting the uer to choose to whether plot is shown
user_prompt = "\n\tDisplay Plot for Error History? (Y, n)"
if user_choice(['Y', 'y', 'N', 'n'], ['Y', 'y'], user_prompt):
    plot_history(error_log, phrasebook)

# Prompting the user to begin chatting
input("\n\tTHE COMPUTER IS READY TO CHAT... (Press Return to Continue)\n")
chatting = True
while chatting:
    # Getting a message from the user
    input_phrase = input("\nChat to your computer >... ")

    # Exiting the chat loop if requested
    if input_phrase == "Stop":
        chatting = False

    # Encoding the users message
    vocab_encoding = input_encoder(input_phrase.lower(), vocabulary)
    print(f"\nYou > {input_phrase}\nInput Encoding: {vocab_encoding}")

    # Generating the computers response
    raw_response = neuralNet.forward(vocab_encoding)
    response = output_decoder(raw_response, phrasebook)
    print(f"\nComputer > {response}\nOutput Encoding: {raw_response}")



