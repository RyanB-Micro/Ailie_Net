
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
phrasebook = ["hello", "my name is Ailie", "hey", "i am Ailie", "bye", "see ya"]




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

error = cost(prediction, targets[0])
print("Error: ", error)
deriv_error = cost_prime(prediction, targets[0])
print("Deriv Error: ", deriv_error)


for itt in range(0, epochs):
    print("\nIteration: ", itt)
    epoch_error = 0
    for sample, target in zip(inputs, targets):
        prediction = neuralNet.forward(sample)
        print("Prediction: ", prediction)
        print("Target: ", target)
        error = cost(prediction, target)
        epoch_error += error
        deriv_error = cost_prime(prediction, target)
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
plt.legend(phrasebook, loc="upper right")
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


input("\n\tTHE COMPUTER IS READY TO CHAT... (Press Return to Continue)\n")
chatting = True

while chatting:
    input_phrase = input("\nChat to your computer >... ")
    if input_phrase == "Stop":
        chatting = False

    vocab_encoding = input_encoder(input_phrase, vocab)
    print(type(vocab_encoding))
    print(f"You: {input_phrase}\n{vocab_encoding}")

    response = neuralNet.forward(vocab_encoding)
    print(output_decoder(response, phrasebook))



