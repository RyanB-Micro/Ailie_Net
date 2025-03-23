
"""
This class represents the layer types to be used within an Ailie Neural Network.

Author: Ryan Brown
Date Created: 17.03.2025
------------------------
Last Edited: 17.03.2025
Last Change: Adding Doc-Strings to internal methods
Repo: https://github.com/CMOSSE101/Ailie_Net
"""
__author__ = "CMOSSE101"

import numpy as np
from .activations import *


class Dense:
    """ Fully Connected Dense Neural Layer

    This class templates a fully connected Neural Layer to be used
    within an Ailie Neural Network.

    The user can specify the number of parallel neurons creat within this
    layer, and the quantity of inouts these neurons use.

    The dimentions and starting values of weight and bias matricies are auto generated.

    :param neurons: The quantity of neurons in this layer.
    :param input_size: The number of weighted connections to the previous layer.
    :type neurons: int.
    :type input_size: int.

    :var self.weight: 2D Matrix to store layer weight values.
    :var self.bias: 1D Matrix to store bias values.
    :var self.weighted_sum: 1D Matrix to store weighted sum result.
    :var self.outputs: 1D Matrix to store and return layer result.

    :Example:
    >>> layer0 = Dense(8, 11)
    Create a layer with 8 neurons, each taking 11 internally weighted inputs.

    """

    def __init__(self, neurons: int, input_size: int):
        self.neurons = neurons
        self.input_size = input_size
        self.weight = np.random.random((neurons, input_size)) - 0.5
        self.bias = np.random.random(neurons) - 0.5
        self.weighted_sum = np.zeros(neurons)
        self.inputs = np.zeros(input_size)
        self.outputs = np.zeros(neurons)
        self.dcost_dweight = None
        self.dcost_dlayer = None
        self.dcost_dbias = None
        self.dcost_dinput = None

    def size(self) -> None:
        """ Displays the current matrix sizes used by the layer """
        print("\nWeight Size: ", self.weight.shape)
        print("Bias Size: ", self.bias.shape)
        print("Input Size: ", self.inputs.shape)
        print("Weighted Sum Size: ", self.weighted_sum.shape)
        print("Output Size: ", self.outputs.shape)

    def vals(self) -> None:
        """ Displays the current values of matrices used within the layer """
        print("\nWeight Value: ", self.weight)
        print("Bias Value: ", self.bias)
        print("Input Value: ", self.inputs)
        print("Weighted Sum Value: ", self.weighted_sum)
        print("Output Value: ", self.outputs)

    def forward(self, input_in: np.ndarray) -> np.ndarray:
        """ Forward propagates the input through the Neural Layer
        :param input_in: The input matrix provided to the entire layer
        :type: input_in: np.ndarray

        :var self.inputs: 1D Matrix to store provided layer input values
        :var self.weighted_sum: 1D Matrix to store results of weighted sum computation
        :var self.outputs: 1D Matrix to return the layers result post activation function
        """
        self.inputs = input_in
        self.weighted_sum = np.dot(self.weight, self.inputs) + self.bias
        # self.outputs = relu(self.weighted_sum)
        self.outputs = self.weighted_sum
        return self.outputs

    def back_stat(self) -> None:
        """ Displays the current values of derivative parameters """
        print("\nDif Layer", self.dcost_dlayer)
        print("Dif Weight", self.dcost_dweight)
        print("Dif Bias", self.dcost_dbias)
        print("Dif Input", self.dcost_dinput)

    def backward(self, layer_error: np.ndarray) -> np.ndarray:
        """

        :param layer_error: Takes in the error attributed to the current layer.

        :var self.dcost_dinput: Stores the derivative of the weight matrix.


        :return: Returns the calculated errors attributed to the previous layer.
        """
        self.dcost_dweight = np.ones((self.neurons, self.input_size))

        self.dcost_dlayer = layer_error # * relu_prime(self.outputs)
        for i in range(0, self.neurons):
            #self.dcost_dweight[i] = self.inputs * layer_error[i]
            self.dcost_dweight[i] = np.dot(self.inputs, layer_error[i])
        # self.dcost_dweight = self.inputs.dot(layer_error)
        self.dcost_dbias = layer_error * 1
        # self.dcost_dinput = self.weight.T * layer_error
        self.dcost_dinput = self.weight.T.dot(layer_error)
        return self.dcost_dinput

    def update(self, alpha: float) -> None:
        self.weight = self.weight - (alpha * self.dcost_dweight)
        self.bias = self.bias - (alpha * self.dcost_dbias)


class Sigmoid_Layer:

    # def __init__(self):
    def forward(self, input_in: np.ndarray) -> np.ndarray:
        """ Forward propagates the input through the Neural Layer
        :param input_in: The input matrix provided to the entire layer
        :type: input_in: np.ndarray

        :var self.inputs: 1D Matrix to store provided layer input values
        :var self.weighted_sum: 1D Matrix to store results of weighted sum computation
        :var self.outputs: 1D Matrix to return the layers result post activation function
        """
        self.inputs = input_in
        self.outputs = sigmoid(self.inputs)
        return self.outputs


    def backward(self, layer_error):
        return sigmoid_prime(self.inputs) * layer_error

    def update(self, alpha: float) -> None:
        # Nothing to change in this layer
        pass


class ReLU_Layer:

    # def __init__(self):
    def forward(self, input_in: np.ndarray) -> np.ndarray:
        """ Forward propagates the input through the Neural Layer
        :param input_in: The input matrix provided to the entire layer
        :type: input_in: np.ndarray

        :var self.inputs: 1D Matrix to store provided layer input values
        :var self.weighted_sum: 1D Matrix to store results of weighted sum computation
        :var self.outputs: 1D Matrix to return the layers result post activation function
        """
        self.inputs = input_in
        self.outputs = relu(self.inputs)
        return self.outputs


    def backward(self, layer_error):
        return relu_prime(self.inputs) * layer_error

    def update(self, alpha: float) -> None:
        # Nothing to change in this layer
        pass