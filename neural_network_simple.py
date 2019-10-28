## Acknowledment code taken from https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
labels = np.array([[1, 0, 0, 1, 1]])
labels = labels.reshape(5, 1)

np.random.seed(42)
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
lr = 0.05

for epoch in range(20000):
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    # feedforward step2
    z = sigmoid(XW)

    # backpropagation step 1
    error = z - labels

    print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

single_point = np.array([1,0,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)
single_point = np.array([0,1,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)
