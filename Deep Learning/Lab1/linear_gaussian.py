#gaussian function
from numpy.linalg import *
import numpy as np
import math

def gaussian_function_0(x):

    return 1/((2*math.pi*np.around(det(v0),decimals=5))**0.5)*np.exp(-np.dot(np.dot((x-u0).T,np.linalg.inv(v0)),x-u0)/2)

def gaussian_function_1(x):

    return 1/((2*math.pi*np.around(det(v1),decimals=5))**0.5)*np.exp(-np.dot(np.dot((x-u1).T,np.linalg.inv(v1)),x-u1)/2)

#generate function
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

x, y = generate_linear()
# average and cov
n = len(x)
x0 = [0,0]
pv0 = []
n0 = 0
x1 = [0,0]
pv1 = []
n1 = 0
for i in range(n):
    if y[i]==0:
        x0 += x[i]
        pv0.append([x[i][0],x[i][1]])
        n0 += 1
    else:
        x1 += x[i]
        pv1.append([x[i][0],x[i][1]])
        n1 += 1

pv0 = np.around(pv0,decimals=5)
pv1 = np.around(pv1,decimals=5)
v0 = np.cov(pv0.T)*(n0-1)/n0
v1 = np.cov(pv1.T)*(n1-1)/n1
u0 = x0/n0
u1 = x1/n1

#test data
testx, testy = generate_linear()

#prediction
s = 0
pred_y = []
for i in range(100):
    p = gaussian_function_0(testx[i])*n0 / (gaussian_function_0(testx[i])*n0 + gaussian_function_1(testx[i])*n1)
    if p > 0.5:
        predy = 0
        pred_y.append(0)
    else:
        predy = 1
        pred_y.append(1)
    if predy == testy[i]:
        s+=1
pred_y = np.array(pred_y).reshape(100,1)

#result
print("LOSS: "+str(100-s))

def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize=18)
    for i in range(testx.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(testx.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

show_result(testx, testy, pred_y)