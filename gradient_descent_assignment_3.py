import datas as datas
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

LEARNING_RATE = 0.001

ITERATION = 1000
initial_a = 0
initial_b = 0
initial_c = 0


# EQUATION = a + b * x + c * x ** 2
# DIFFERENTIAL_EQUATION = b + 2 * C * x
def gradient_descent_runner(_a, _b, _c, _x_array,_y_array):
    cost = 0.0
    i = 0
    a = _a
    b = _b
    c = _c

    for x in range(0, ITERATION):
        a, b, c = gradient_descent_step(a, b, c, _x_array, _y_array, cost)
        print('gradient_descent_runner Epoch ' + str(x) + ' _a_ ' + str(a) + ', b_' + str(b) + ', c_' + str(c))
    # print(a, b, c)
    return [a, b, c]


def gradient_descent_step(_a, _b, _c, _x_array, _y_array, _output_y):
    cost = 0.0
    i = 0
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0

    i = 0
    length = len(_x_array)
    # find gradient of the cost function
    for x in _x_array:
        cost = (-float(_a + _b * x + _c * x ** 2) + float(_y_array[i]))
        print('cost '+str(cost))
        # print('gradient_descent_step_ cost ' + str(_output_y))
        # a_gradient += + (2 / length) * (_y_array[i] - cost) * (-1)
        # b_gradient += + (2 / length) * (_y_array[i] - cost) * (-_x_array[i])
        # c_gradient += + (2 / length) * (_y_array[i] - cost) * (-2 * _x_array[i])
        # a_gradient += (2 / length) * (_y_array[i] - _output_y) * (-1)
        # b_gradient += (2 / length) * (_y_array[i] - _output_y) * (-_x_array[i])
        # c_gradient += (2 / length) * (_y_array[i] - _output_y) * (-2 * _x_array[i])
        a_gradient += (2 / length) * cost * (-1)
        b_gradient += (2 / length) * cost * (-x)
        c_gradient += (2 / length) * cost * (-2 * x)

        i = i + 1
    print('a_gradient '+str(a_gradient)+' b_gradient '+str(b_gradient)+' c_gradient '+str(c_gradient))
    # update values
    new_a = _a - LEARNING_RATE * a_gradient
    new_b = _b - LEARNING_RATE * b_gradient
    new_c = _c - LEARNING_RATE * c_gradient
    return [new_a, new_b, new_c]


def predict(_a, _b, _c, test):
    print(_a)
    print(_b)
    print(_c)
    print(test)
    val = float(_a + _b * test + _c * (test ** 2))
    print(val)
    return val


datas = pd.read_csv('/home/rajroy/Desktop/sample_assisgnment.csv', delimiter=',')
data_array = datas
x = datas.iloc[:, 0].values
y = datas.iloc[:, 1].values
print(x)
print(y)
a, b, c = 0, 0, 0
a, b, c = gradient_descent_runner(initial_a, initial_b, initial_c, x, y)
print(a)
print(predict(a, b, c, 1.83))
