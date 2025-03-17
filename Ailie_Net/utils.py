import Ailie_Net

def cost(prediction, target):
    return (target - prediction)**2

def cost_prime(prediction, target):
    return 2*(prediction - target)