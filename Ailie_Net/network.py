
import numpy as np
from .activations import *

class AilieNet():
    def __init__(self, ):
        self.layers = []

    def forward(self, input_in):
        self.input_buf = input_in
        for layer in self.layers:
            self.prediction_buf = layer.forward(self.input_buf)
            self.input_buf = self.prediction_buf
        # final activation
        #self.prediction_buf = sigmoid(self.prediction_buf)
        return self.prediction_buf

    def backward(self, error, learn_rate):
        self.layer_error = error
        for layer in reversed(self.layers):
            self.inner_error = layer.backward(self.layer_error)
            self.layer_error = self.inner_error
            layer.update(learn_rate)
        return self.layer_error

    def add(self, layer):
        self.layers.append(layer)