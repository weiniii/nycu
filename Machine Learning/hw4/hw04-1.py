import numpy as np
import matplotlib.pyplot as plt

def gaussian(m, s):
    
    sample = np.sum(np.random.uniform(0, 1, 12)) - 6
    output = m + sample * s ** (1 / 2)
    
    return output

def generator_data(mx, vx, my, vy, n):
    
    D = np.ones((3, n))

    x = []
    y = []
    for i in range(n):
        x.append(gaussian(mx, vx))
        y.append(gaussian(my, vy))
    
    D[1] = np.array(x)
    D[2] = np.array(y)
    
    return D

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def derivate_sigmoid(s):
    return np.multiply(s, 1.0 - s)

def sigmoid_newton(z):
    return np.exp(-z) / (1.0 + np.exp(-z)) ** 2

def statistic(i, correct, confusion, sensitivity, N):
    
    confusion[i, 0, 0] = np.sum(correct[:N])
    confusion[i, 0, 1] = N - np.sum(correct[:N])
    confusion[i, 1, 1] = np.sum(correct[N:])
    confusion[i, 1, 0] = N - np.sum(correct[N:])


    sensitivity[i, 0] = confusion[i, 0, 0] / N
    sensitivity[i, 1] = confusion[i, 1, 1] / N
    
    return confusion, sensitivity

def show(i, weight, confusion, sensitivity):
    if i == 0:
        print('Gradient Descent')
    else:
        print('Newton\'s Method')
        
    print('')
    print('w:')
    for ind in range(3):
        print(f'   {weight[ind]: 4.11f}')
    
    title = ['', 'Predict cluster 1', 'Predict cluster 2']
    columns = ['Is cluster 1', 'Is cluster 2']
    
    print('')
    
    print('Confusion Matrix:')
    print(f'{title[0]:^10}     {title[1]:^15}     {title[2]:^15}')
    for ind in range(2):
        print(f'{columns[ind]:^10}     {confusion[i, ind, 0]:^15}    {confusion[i, ind, 1]:^15}')
    
    print('')
    for ind in range(2):
        print(f'Sensitivity (Successfully predict cluster {ind + 1}):  {sensitivity[i, ind]:1.4f}')

def plot(D, c1, c2, c3):
    
    plt.subplot(1,3,1)
    plt.title('Ground Truth')
    plt.scatter(D[1], D[2], c=c1)

    plt.subplot(1,3,2)
    plt.title('Gradient Descent')
    plt.scatter(D[1], D[2], c=c2)

    plt.subplot(1,3,3)
    plt.title('Newton\'s Method')
    plt.scatter(D[1], D[2], c=c3)
    
    plt.show()

N = 50
learning_rate = 1e-3

# case 1
mean_x1 = 1
var_x1 = 2
mean_y1 = 1
var_y1  = 2

mean_x2 = 10
var_x2 = 2
mean_y2 = 10
var_y2 = 2

# case 2
mean_x1 = 1
var_x1 = 2
mean_y1 = 1
var_y1  = 2

mean_x2 = 3
var_x2 = 4
mean_y2 = 3
var_y2 = 4

D1 = generator_data(mean_x1, var_x1, mean_y1, var_y1, N)
D2 = generator_data(mean_x2, var_x2, mean_y2, var_y2, N)
D = np.concatenate((D1, D2), axis=1)
Y = np.array([0 for i in range(N)] + [1 for i in range(N)])
color_y = np.where(Y == 0, 'r', 'b')

# Gradient Descent
w = np.ones(3)
for i in range(10000):
    prediction = sigmoid(w @ D)
    loss = prediction - Y
    delta = loss * derivate_sigmoid(prediction)
    w -= learning_rate * D @ delta

y_hat = np.where(prediction < 0.5, 0, 1)
color = np.where(y_hat == 0, 'r', 'b')

correct = np.where(Y == y_hat, 1, 0)

confusion = np.empty((2, 2, 2)).astype(int)
sensitivity = np.empty((2, 2))

confusion, sensitivity = statistic(0, correct, confusion, sensitivity, N)

# Newton's Method
X_n = np.zeros((3))
D_matrix = np.eye(2 * N) * sigmoid_newton(X_n @ D)
AtA = D @ D_matrix @ D.T
H = AtA + 1e-3 * np.eye(3)
inv_H = np.linalg.inv(H)
delta_f = D @ (Y - sigmoid(X_n @ D))
X_n = inv_H @ delta_f


newton_prediction = sigmoid(X_n @ D)
newton_y_hat = np.where(newton_prediction < 0.5, 0, 1)
newton_color = np.where(newton_y_hat == 0, 'r', 'b')

correct = np.where(newton_y_hat == Y, 1, 0)
confusion, sensitivity = statistic(1, correct, confusion, sensitivity, N)

show(0, w, confusion, sensitivity)
print('')
print('-' * 50)
show(1, X_n, confusion, sensitivity)

plot(D, color_y, color, newton_color)
