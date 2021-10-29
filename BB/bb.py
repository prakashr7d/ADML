import numpy as np
from numpy import random

from numpy import dot


def sigmoid_derivation(sop):
    return 1 / 1 + np.exp(-1 * sop)


def error_for(predicted, target):
    return np.power(predicted - target, 2)


def error_predicted_derivation(predicted, target):
    return 2 * (predicted - target)


def sigmoid_sop_derivation(sop):
    return sigmoid_derivation(sop) * (1.0 - sigmoid_derivation(sop))


def sop_weight_derivation(x):
    return x


def update_weight(w, grad, learning_rate):
    return w - learning_rate * grad

target = 0.7
LR = 0.01
x1 = 0.1
x2 = 0.4

inputs_for_the_network = [x1, x2]
predicted_outputs = []
total_error = []
weights = random.rand(2)

for i in range(100):

    print("Epoch", i, ": \n")
    weight1, weight2 = weights[0], weights[1]
    result = dot(weights, inputs_for_the_network)


    print("\tx", result)
    predicted = sigmoid_derivation(result)
    print("It predicted", predicted)


    err = error_for(target, predicted)
    print("It error", err)


    predicted_outputs.append(predicted)

    g1 = error_predicted_derivation(predicted, target)
    g2 = sigmoid_sop_derivation(result)
    g3w1 = sop_weight_derivation(x1)
    g3w2 = sop_weight_derivation(x2)

    gradw1 = g3w1 * g2 * g1
    gradw2 = g3w2 * g2 * g1

    W1 = update_weight(weight1, gradw1, LR)
    W2 = update_weight(weight2, gradw2, LR)

    weights = [weight1, weight2]
    print("It updated weights:", weights, "In")
