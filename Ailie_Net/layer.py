class Dense():
    def __init__(self, neurons, input_size):
        self.neurons = neurons
        self.input_size = input_size
        self.weight = np.random.random((neurons, input_size)) - 0.5
        self.bias = np.random.random(neurons) - 0.5
        self.weighted_sum = np.zeros(neurons)
        self.inputs = np.zeros(input_size)
        self.outputs = np.zeros(neurons)

    def size(self):
        print()
        print("Weight Size: ", self.weight.shape)
        print("Bias Size: ", self.bias.shape)
        print("Input Size: ", self.inputs.shape)
        print("Weighted Sum Size: ", self.weighted_sum.shape)
        print("Output Size: ", self.outputs.shape)

    def vals(self):
        print()
        print("Weight Value: ", self.weight)
        print("Bias Value: ", self.bias)
        print("Input Value: ", self.inputs)
        print("Weighted Sum Value: ", self.weighted_sum)
        print("Output Value: ", self.outputs)

    def forward(self, input_in):
        self.inputs = input_in
        self.weighted_sum = np.dot(self.weight, self.inputs) + self.bias
        self.outputs = relu(self.weighted_sum)
        return self.outputs

    def back_stat(self):
        print()
        print("Dif Layer", self.dcost_dlayer)
        print("Dif Weight", self.dcost_dweight)
        print("Dif Bias", self.dcost_dbias)
        print("Dif Input", self.dcost_dinput)

    def backward(self, layer_error):
        self.dcost_dweight = np.ones((self.neurons, self.input_size))

        self.dcost_dlayer = layer_error
        for i in range(0, self.neurons):
            self.dcost_dweight[i] = self.inputs * layer_error[i]
        # self.dcost_dweight = self.inputs.dot(layer_error)
        self.dcost_dbias = layer_error * 1
        # self.dcost_dinput = self.weight.T * layer_error
        self.dcost_dinput = self.weight.T.dot(layer_error)
        return self.dcost_dinput

    def update(self, alpha):
        # print()
        # print("\nWeight: ", self.weight)
        # print("Weight Cost: ", self.dcost_dweight)
        self.weight = self.weight - (alpha * self.dcost_dweight)
        self.bias = self.bias - (alpha * self.dcost_dbias)