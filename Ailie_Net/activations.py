
def sigmoid(l):
    return 1 / (1 + np.exp(-l))

def sigmoid_prime(l):
    sigmoid(l) * (1 - sigmoid(l))

def relu(l):
    return np.maximum(0, l)

def relu_prime(l):
    return np.where(l > 0, 1, 0)