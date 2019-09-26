import pandas as pd
import matplotlib.pyplot as plt
import math
import csv

LEARNING_RATE = 0.0001

ITERATION = 1000
initial_a = 1
initial_b = 0
initial_c = 0
initial_d = 0
initial_e = 0


# EQUATION = a + b * x + c * x ** 2
def gradient_descent_runner(_a, _b, _c, _d, _e, df):
    cost = 0.0
    a = _a
    b = _b
    c = _c
    d = _d
    e = _e

    for x in range(0, ITERATION):
        a, b, c, d, e = gradient_descent_step(a, b, c, d, e, df)
    return [a, b, c, d, e]


def get_cost(_a, _b, _c, _d, _e, x):
    cost = 1 / (1 + math.exp(-1 * (_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])))
    print('Cost_'+str(cost))
    cost = x[4] - cost
    return cost


def gradient_descent_step(_a, _b, _c, _d, _e, df):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0
    d_gradient = 0
    e_gradient = 0
    i = 0
    length = len(df)
    # find gradient of the cost function
    # cost = get_cost(_a, _b, _c, _d, _e, x)
    for x in df:
        cost = float(get_cost(_a, _b, _c, _d, _e, x))
        # print(cost)

        # print(cost)
        a_gradient += (1 / length) * cost
        b_gradient += (1 / length) * cost * (x[0])
        c_gradient += (1 / length) * cost * (x[1])
        d_gradient += (1 / length) * cost * (x[2])
        e_gradient += (1 / length) * cost * (x[3])

        i = i + 1
    # print('a_gradient ' + str(a_gradient) + ' b_gradient ' + str(b_gradient) + ' c_gradient ' + str(c_gradient))
    # update values
    new_a = _a - LEARNING_RATE * a_gradient
    new_b = _b - LEARNING_RATE * b_gradient
    new_c = _c - LEARNING_RATE * c_gradient
    new_d = _d - LEARNING_RATE * d_gradient
    new_e = _e - LEARNING_RATE * d_gradient
    return [new_a, new_b, new_c, new_d, new_e]


def predict(_a, _b, _c, _d, _e, x):
    val = float(_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])
    return val


def accuracy(_a, _b, _c, _d, _e, array):
    total_accuracy = 0
    result_array = []
    for x in array:
        val = float(_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])
        if val > 0.5:
            result_array.append(1)
        else:
            result_array.append(0)
    i = 0

    for res in result_array:
        if res == array[i][4]:
            total_accuracy += 1
        i = i + 1
    return (total_accuracy * 100) / len(result_array)


def data_reader():
    data_dir = '/home/rajroy/Desktop/iris.data'
    data_array = []
    with open(data_dir) as csv_file:
        reader = csv.reader(csv_file)  # change contents to floats
        for row in reader:  # each row is a list
            value = 0
            if row[4] == 'Iris-setosa':
                value = 0
            elif row[4] == 'Iris-versicolor':
                value = 1
            data_array.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), value])

    return data_array


data_array = data_reader()
a, b, c, d, e = 0, 0, 0, 0, 0
a, b, c, d, e = gradient_descent_runner(initial_a, initial_b, initial_c, initial_d, initial_e, data_array)

print(a, b, c, d, e)
print(data_array[22])
print(predict(a, b, c, d, e, data_array[22]))
# print('accuracy ' + str(accuracy(a, b, c, d, e, data_array)))
#
# plt.scatter(x, y, color='blue')
#
# plt.plot(x, predict(a, b, c, x), color='red')
#
# plt.title('Assignment 3 - programming 1_ using gradient_descent')
# plt.ylabel('Height')
# plt.xlabel('Weight')
#
# plt.show()
