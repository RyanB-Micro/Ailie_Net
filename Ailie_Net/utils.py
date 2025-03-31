import Ailie_Net
# import math
import numpy as np

#def cost(prediction, target):
def squared_error(prediction, target):
    return (target - prediction)**2

def squared_error_prime(prediction, target):
#def cost_prime(prediction, target):
    return 2*(prediction - target)

def cross_entropy_error(prediction, target):
    error = np.zeros(len(prediction))
    for i, element in enumerate(prediction):
        #error[i] = -(target[i]*np.log(prediction[i]) + (1-target[i])*np.log(1-prediction[i]))
        error[i] = -(target[i]*np.log(prediction[i]) + (1-target[i])*np.log(1-prediction[i]))
    return error

def user_choice(options_list, actionable_choices, user_prompt):
    user_choice = ""
    while user_choice not in options_list:
        user_choice = input(user_prompt)
        return user_choice in actionable_choices