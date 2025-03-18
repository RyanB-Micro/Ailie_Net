import Ailie_Net

def cost(prediction, target):
    return (target - prediction)**2

def cost_prime(prediction, target):
    return 2*(prediction - target)

def user_choice(options_list, actionable_choices, user_prompt):
    user_choice = ""
    while user_choice not in options_list:
        user_choice = input(user_prompt)
        return user_choice in actionable_choices