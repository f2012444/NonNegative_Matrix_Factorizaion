from numpy import array
from numpy import transpose
from numpy.matrixlib.defmatrix import matrix
import numpy as np
import numpy
import re
import os
import matplotlib.pyplot as pyplot

print os.getcwd()


def get_cost(a, b):
    cost = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            cost += pow(a[i, j] - b[i, j], 2)
    return cost


def get_matrix_factors(data_matrix, rows, columns, num_of_factors, num_of_iterations, epsilon):
    # If any value of matrix less than zero, return Not possible to get non-negative matrix factors
    for x in range(rows):
        for y in range(columns):
            num = data_matrix[x][y]
            if num < 0:
                raise ValueError('Not possible to get non-negative matrix factors')

    w = matrix([[np.random.uniform(2.0, 10.0) for i in range(num_of_factors)]
                for j in range(rows)])
    h = matrix([[np.random.uniform(2.0, 10.0) for i in range(columns)]
                for j in range(num_of_factors)])
    init_cost = get_cost(data_matrix, w*h)
    print("init cost is", init_cost)
    for i in range(num_of_iterations):
        curr_cost = get_cost(data_matrix, w * h)
        print(" cost is", i, curr_cost)
        if curr_cost <= epsilon:
            return w, h
        h1 = (transpose(w) * data_matrix)
        h2 = (transpose(w) * w * h)
        h = matrix(array(h) * array(h1) / array(h2))
        w1 = (data_matrix * transpose(h))
        w2 = (w * h * transpose(h))
        w = matrix(array(w) * array(w1) / array(w2))
    return w, h

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


wd = os.getcwd() + '/orl_faces'
for i in range(1, 41):
    cur_dir = '/s'+str(i)
    for j in range(1, 2):
        pwd = wd + cur_dir + '/' + str(j) + '.pgm'
        vals = read_pgm(pwd)
        #temp = sum(vals)
        image, image2 = get_matrix_factors(vals, 112, 92, num_of_factors=50, num_of_iterations=1000, epsilon=0)

        pyplot.imshow(image*image2, pyplot.cm.gray)
        pyplot.show()





def get_matrix_factors(data_matrix, rows, columns, num_of_factors, num_of_iterations, epsilon):
    # If any value of matrix less than zero, return Not possible to get non-negative matrix factors
    for x in range(rows):
        for y in range(columns):
            num = Matrix[x][y]
            if num < 0:
                raise ValueError('Not possible to get non-negative matrix factors')

    w = matrix([[np.random.uniform(2.0, 10.0) for i in range(num_of_factors)]
                for j in range(rows)])
    h = matrix([[np.random.uniform(2.0, 10.0) for i in range(columns)]
                for j in range(num_of_factors)])
    init_cost = get_cost(data_matrix, w*h)
    print("init cost is", init_cost)
    for i in range(num_of_iterations):
        curr_cost = get_cost(data_matrix, w * h)
        print(" cost is", i, curr_cost)
        if curr_cost <= epsilon:
            return w, h
        h1 = (transpose(w) * data_matrix)
        h2 = (transpose(w) * w * h)
        h = matrix(array(h) * array(h1) / array(h2))
        w1 = (data_matrix * transpose(h))
        w2 = (w * h * transpose(h))
        w = matrix(array(w) * array(w1) / array(w2))
    return w, h
def generalized_KL(x, y, eps=1.e-8, axis=None):
    return (np.multiply(x, np.log(np.divide(x + eps, y + eps))) - x + y
            ).sum(axis=axis)
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))



rows = 100
columns = 500
Matrix = np.zeros((rows, columns))
w = matrix([[np.random.uniform(2.0, 10.0) for i in range(50)]
                for j in range(rows)])
h = matrix([[np.random.uniform(2.0, 10.0) for i in range(columns)]
                for j in range(50)])

#Matrix = [[0 for x in range(columns)] for y in range(rows)]
for x in range(rows):
    for y in range(columns):
        Matrix[x][y] = np.random.uniform(2.0, 10.0)
cost_kl = generalized_KL(Matrix, w*h)
print("kl cost is", cost_kl)
w, h = get_matrix_factors(Matrix, rows, columns, num_of_factors=50, num_of_iterations=1000, epsilon = 0)

cost = get_cost(Matrix, w*h)
print("final cost is ", cost)




