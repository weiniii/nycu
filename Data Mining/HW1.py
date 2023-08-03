"""
    NYCU 2023 Spring Data Mining Homework 1
"""
import pandas as pd
import numpy as np

global give_up
give_up = []

def get_data():
    Is_normalize = True
    h = 9
    train_df = pd.read_csv(r'train.csv')
    test_df = pd.read_csv(r'test_X.csv', header = None)
    train_data = train_df.values[:, 3:]
    test_data = test_df.values[:, 2:]
    
    def fillup(d, mode):
        remove = []
        if mode == 'Normal':
            for i in range(len(d)):
                now_data = []
                loss_data = []
                
                for j in range(len(d[i])):
                    try:
                        d[i, j] = float(d[i, j])
                        now_data.append(d[i, j])
                    except:
                        loss_data.append(j)
                if loss_data and now_data:
                    mu = np.array(now_data).mean()
                    std = np.array(now_data).std()
                    for l in loss_data:
                        d[i, l] = np.random.normal(mu, std, 1).item()
                        # d[i, l] = mu
                elif loss_data:
                    remove.append(i)
                    d[i] = np.zeros(len(d[i]))
                else:
                    pass
        elif mode == 'Zero':
            for i in range(d.shape[0]):
                loss_data = []
                for j in range(d.shape[1]):
                    try:
                        d[i, j] = float(d[i, j])
                    except:
                        d[i, j] = float(0)
                        loss_data.append(j)
                if len(loss_data) == d.shape[1]:
                    remove.append(i)
        else:
            raise ValueError
        
        remove = [int(x / 18) for x in remove]
        return d, remove
    
    train_data, remove = fillup(train_data, 'Normal')
    test_data, _ = fillup(test_data, 'Normal')

    reshape_train_data = np.empty((18, int(len(train_data) / 18) * 24))
    for d in range(int(len(train_data) / 18)):
        reshape_train_data[ : , d * 24 : (d + 1) * 24] = train_data[d * 18 : (d + 1) * 18, : ]
    reshape_test_data = np.empty((18, int(len(test_data)/18) * 9))
    for d in range(int(len(test_data) / 18)):
        reshape_test_data[ : , d * 9: (d + 1) * 9] = test_data[d * 18 : (d + 1) * 18, :]

    if give_up:
        count = sum(np.array(give_up) < 9)
        reshape_train_data = np.delete(reshape_train_data, give_up, axis=0)
        reshape_test_data = np.delete(reshape_test_data, give_up, axis=0)
    else:
        pass

    if len(reshape_test_data) != len(reshape_train_data):
        raise IndexError
    else:
        f = len(reshape_test_data)

    test_X = np.empty((int(len(test_data)/18), 9 * f))

    for i in range(len(test_X)):
        test_X[i] = reshape_test_data[ : , i * 9 : (i + 1) * 9].reshape(1, -1)

    X = np.empty((5652, f * h))
    Y = np.empty((5652, 1))

    try:
        label = 9 - count
    except:
        label = 9
        
    for i in range(len(X)):
        m = int(i/471)
        X[i] = reshape_train_data[ : , i + m * 9  : i + h + m * 9].reshape(-1)
        Y[i] = reshape_train_data[label, i + h + m * 9]

    def Normalize(d1, d2):
        mean_x = np.mean(d1, axis = 0)
        std_x = np.std(d1, axis = 0) 
        for i in range(d1.shape[0]):
            for j in range(d1.shape[1]):
                if std_x[j] != 0:
                    d1[i][j] = (d1[i][j] - mean_x[j]) / std_x[j]
        
        for i in range(d2.shape[0]):
            for j in range(d2.shape[1]):
                if std_x[j] != 0:
                    d2[i][j] = (d2[i][j] - mean_x[j]) / std_x[j]

        return d1, d2
            
    if Is_normalize:
        X, test_X = Normalize(X, test_X)
    else:
        pass
    
    return X, Y, test_X, remove

def suffle(rate, Is):
    x, y, test, remove  = get_data()

    remove = [j + (int(d % 20) - 1) * 24 + int(d / 20) * 471 for d in remove for j in range(-8, 24)]
    remove = list(set(remove))
    
    ind = list(np.random.choice(471, int(471 * rate), replace=False))
    for i in range(1, 12):
        ind += list(np.random.choice(471, int(471 * rate), replace=False) + 471 * i )

    if Is:
        ind = list(set(ind) - set(remove))
    else:
      pass

    train_x = x[ind]
    train_y = y[ind]
    test_x = np.delete(x, ind, axis=0)
    test_y = np.delete(y, ind, axis=0)

    return train_x, train_y, test_x, test_y

def result(dx, dy, weight):
    dx = np.concatenate((np.ones([len(dx), 1]), dx), axis = 1).astype(float)
    pred = np.dot(dx, weight)
    return np.sqrt(sum(np.power(dy - pred, 2))/len(dy))

def prediction(test, w):
    test = np.concatenate((np.ones([len(test), 1]), test), axis = 1).astype(float)

    pred_test = np.dot(test, w)
    final = np.empty((len(pred_test), 2)).astype(object)
    for i in range(len(final)):
        final[i, 0] = 'index_'+str(i)
        final[i, 1] = str(pred_test[i][0])

    df = pd.DataFrame(final, columns = ['index','answer'])
    df.to_csv('STUDENT_ID.csv', index=None)
    
give_up = [0, 11, 3, 17, 16, 5, 4, 13]
x, y, test, remove  = get_data()

remove = [j + (int(d % 20) - 1) * 24 + int(d / 20) * 471 for d in remove for j in range(-8, 24)]
remove = list(set(remove))
x = np.delete(x, remove, axis=0)
y = np.delete(y, remove, axis=0)

feature = x.shape[0]
rate = 0.5

n = int(len(x)/12)
train_x = x[ : int(n * rate)]
train_y = y[ : int(n * rate)]
test_x = x[int(n * rate) : n]
test_y = y[int(n * rate) : n]
for i in range(1,12):
    train_x = np.concatenate((train_x, x[int(n * i) : int(n * i + n * rate)]))
    train_y = np.concatenate((train_y, y[int(n * i) : int(n * i + n * rate)]))
    try:
        test_x = np.concatenate((test_x, x[int(n * i + n * rate) : int(n * (i + 1))]))
        test_y = np.concatenate((test_y, y[int(n * i + n * rate) : int(n * (i + 1))]))
    except:
        pass

train_x_plus = np.concatenate((np.ones([len(train_x), 1]), train_x), axis = 1).astype(float)
feature = len(x[0])
w = np.zeros([feature + 1, 1])
gradient = np.zeros([feature + 1, 1])
iter_time = 5000
rang = int(iter_time / 10)
learning_rate = 0.9

## Adam optimzier ##
beta1 = 0.999
beta2 = 0.9999
m1 = np.zeros([feature + 1, 1])
v1 = np.zeros([feature + 1, 1])
m_hat = np.zeros([feature + 1, 1])
v_hat = np.zeros([feature + 1, 1])
eps = 10e-8
## Adam optimzier ##
print(f'Start the {iter_time} times training!!!')
train_loss = []
test_loss = []
for t in range(iter_time):
    train_x_plus = np.concatenate((np.ones([len(train_x), 1]), train_x), axis = 1).astype(float)
    
    loss = np.sqrt(np.sum(np.power(np.dot(train_x_plus, w) - train_y, 2))/len(train_x_plus))
    gradient = 2 * np.dot(train_x_plus.T, np.dot(train_x_plus, w) - train_y) / loss / 2

    m1 = beta1 * m1 + (1 - beta1) * gradient
    v1 = beta2 * v1 + (1 - beta2) * (gradient ** 2)
    m_hat = m1 /(1 - beta1)
    v_hat = v1 /(1 - beta2)
    w -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    train_loss.append(loss)
    test_loss.append(result(test_x, test_y, w))
    if t % rang == 0:
        print(f'The {t}/{iter_time} epochs RMSE : {loss}')
        
print(f'The minimum of test loss : {min(test_loss)}')
# prediction(test, w)