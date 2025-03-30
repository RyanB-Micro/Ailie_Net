import sys

# Importing external library for graph capabilities
import matplotlib.pyplot as plt
# Importing external numpy maths library
import numpy as np
# importing the library to read JSON files
import json

import Ailie_Net as ai

def train_network(network, inputs, targets, epochs, learn_rate, error_log):
    for itt in range(0, epochs):
        print("\nIteration: ", itt)
        epoch_error = 0
        sample_count = 0
        progress = []
        for sample, target in zip(inputs, targets):
            prediction = network.forward(sample)

            error = ai.cost(prediction, target)
            epoch_error += error
            deriv_error = ai.cost_prime(prediction, target)
            back_error = network.backward(deriv_error, learn_rate)
            # if (itt % 10 == 0):
            #     print("Prediction: ", prediction)
            #     print("Target: ", target)
            #     print("Error: ", error)
            #else:
            sys.stdout.write(f"\rProgress: {progress} {sample_count}/{len(inputs)}")
            sys.stdout.flush()

            sample_count += 1
            progress = '#' * int(sample_count/100)

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


def simple_hot(inputs, categories):
    encoding_buff = np.zeros((len(inputs), len(categories)))

    for index, sample in enumerate(inputs, start=0):
        for position, cat in enumerate(categories, start=0):
            if sample == cat:
                encoding_buff[index][position] = 1

    return encoding_buff

def hot_decode(input, categories):
    decoding = np.zeros(len(categories))

    for position, cat in enumerate(categories, start=0):
        if input == cat:
            decoding[position] = 1

    return decoding
    #return "Error: Not category match found"



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
    if phrasebook_encoding[chosen_phrase] > 0.6:
        return phrasebook[chosen_phrase]
    else: # No single output was strong enough
        return "Sorry, i do not understand."


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
