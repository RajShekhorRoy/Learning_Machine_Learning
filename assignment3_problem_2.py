import pandas as pd
import matplotlib.pyplot as plt
import math
LEARNING_RATE = 0.0001

ITERATION = 10000
initial_a = 1
initial_b = 0
initial_c = 0
initial_d = 0


# EQUATION = a + b * x + c * x ** 2
def gradient_descent_runner(_a, _b, _c, _d, _e, df):
    cost = 0.0
    a = _a
    b = _b
    c = _c
    d = _d


    for x in range(0, ITERATION):
        a, b, c, d, e = gradient_descent_step(a, b, c, d, e, df)
    return [a, b, c]

def get_cost(_a, _b, _c, _d, _e, df):
    cost = 0
    cost =1 +  math.exp(-_a *2)
    return
def gradient_descent_step(_a, _b, _c, _d,  df):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0
    d_gradient = 0
    e_gradient = 0
    i = 0
    length = len(df)
    # find gradient of the cost function
    cost = get_cost(_a, _b, _c, _d,)
    for x in df:
        cost = get_cost(_a, _b, _c, _d, df)
        a_gradient += (2 / length) * cost * (-1)
        b_gradient += (2 / length) * cost * (-x)
        c_gradient += (2 / length) * cost * (-2 * x)
        d_gradient += (2 / length) * cost * (-2 * x)


        i = i + 1
    # print('a_gradient ' + str(a_gradient) + ' b_gradient ' + str(b_gradient) + ' c_gradient ' + str(c_gradient))
    # update values
    new_a = _a - LEARNING_RATE * a_gradient
    new_b = _b - LEARNING_RATE * b_gradient
    new_c = _c - LEARNING_RATE * c_gradient
    new_d = _d - LEARNING_RATE * d_gradient
    return [new_a, new_b, new_c]


def predict(_a, _b, _c, array):
    result = []
    for x in array:
        val = float(_a + _b * x + _c * (x ** 2))
        result.append(val)
    return result


def accuracy(_a, _b, _c, _x_array, _y_array):
    i = 0
    total_accuracy = 0
    for x in _x_array:
        val = (float(_a + _b * x + _c * (x ** 2))) / _y_array[i]
        per = (1 - (val / _y_array[i]))
        print(per)
        total_accuracy += per
        i = i + 1
    return abs(100 * total_accuracy / len(_x_array))


datas = pd.read_csv('/home/rajroy/Desktop/iris.data', delimiter=',')
data_array = datas
x = datas.iloc[:, ].values
y = datas.iloc[:, 1].values
print(datas.iloc[0:1])
print(y)
a, b, c ,d,e = 0, 0, 0, 0, 0
a, b, c,d,e = gradient_descent_runner(initial_a, initial_b, initial_c, initial_d,data_array)
#
# print(accuracy(a, b, c, x, y))
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
