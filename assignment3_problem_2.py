import math
import csv

LEARNING_RATE = 0.0001
ITERATION = 10000
initial_a = 1  # bias  (for small iterations it gives bad accuracy if it is not 0)
initial_b = 0
initial_c = 0
initial_d = 0
initial_e = 0


def gradient_descent_runner(_a, _b, _c, _d, _e, df):
    a = _a
    b = _b
    c = _c
    d = _d
    e = _e
    for x in range(0, ITERATION):
        a, b, c, d, e = gradient_descent_step(a, b, c, d, e, df)
    return [a, b, c, d, e]


def sigmoid_function(_a, _b, _c, _d, _e, x):
    h = 1 / (1 + math.exp(-1 * (_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])))  # hypothesis
    if h >= 0.5:
        return 1
    elif h < 0.5:
        return 0


def gradient_descent_step(_a, _b, _c, _d, _e, df):
    a_gradient = 1
    b_gradient = 0
    c_gradient = 0
    d_gradient = 0
    e_gradient = 0
    i = 0
    length = len(df)
    for x in df:
        # thete_new :=  theta - alpha*cost
        # for this case cost is summation of the (hypothesis(x(i))-y(i))x(i)
        cost = float(sigmoid_function(_a, _b, _c, _d, _e, x))
        a_gradient += (1 / length) * (cost - x[4])
        b_gradient += (1 / length) * (cost - x[4]) * (x[0])
        c_gradient += (1 / length) * (cost - x[4]) * (x[1])
        d_gradient += (1 / length) * (cost - x[4]) * (x[2])
        e_gradient += (1 / length) * (cost - x[4]) * (x[3])

        i = i + 1
    new_a = _a - LEARNING_RATE * a_gradient
    new_b = _b - LEARNING_RATE * b_gradient
    new_c = _c - LEARNING_RATE * c_gradient
    new_d = _d - LEARNING_RATE * d_gradient
    new_e = _e - LEARNING_RATE * d_gradient
    return [new_a, new_b, new_c, new_d, new_e]


def predict(_a, _b, _c, _d, _e, x):
    val = float(_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])
    cost = 1 / (1 + math.exp(-1 * val))
    return cost


def accuracy(_a, _b, _c, _d, _e, array):
    number_of_one = 0
    correct_0 = 0
    correct_1 = 0
    print('total number of dataset: ' + str(len(array)) + ', Iteration: ' + str(ITERATION) + ', Learning_rate: ' + str(
        LEARNING_RATE))

    for x in array:
        val = float(_a + _b * x[0] + _c * x[1] + _d * x[2] + _e * x[3])
        cost = 1 / (1 + math.exp(-1 * val))

        if cost < 0.5:
            if x[4] == 0:
                correct_0 = correct_0 + 1
        if cost >= 0.5:
            if x[4] == 1:
                correct_1 = correct_1 + 1
        if x[4] == 1:
            number_of_one += 1
    print('true number of 1 i.e Iris-versicolor ' + str(correct_1) + ' out of ' + str(number_of_one))

    print('true number of 0 i.e Iris-setosa     ' + str(correct_0) + ' out of ' + str(len(array) - number_of_one))

    return (100 * float(correct_1 + correct_0)) / len(array)


def data_reader():
    # considers only the 2 values for classification Iris-setosa & Iris-versicolor
    data_dir = '/home/rajroy/Desktop/iris.data'
    data_array = []
    with open(data_dir) as csv_file:
        reader = csv.reader(csv_file)  # change contents to floats
        for row in reader:  # each row is a list
            value = 0
            if len(row) > 1:
                if row[4] == 'Iris-setosa':
                    value = 0
                    data_array.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), value])
                elif row[4] == 'Iris-versicolor':
                    value = 1
                    data_array.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), value])

    return data_array


data_array = data_reader()
a, b, c, d, e = 0, 0, 0, 0, 0
a, b, c, d, e = gradient_descent_runner(initial_a, initial_b, initial_c, initial_d, initial_e, data_array)
test_no = 1
print('Accuracy ' + str(accuracy(a, b, c, d, e, data_array)) + '%')
