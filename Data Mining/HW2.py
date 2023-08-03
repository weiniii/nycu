import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_X.csv')
feature = df.columns
category = [1, 3, 5, 6, 7, 8, 9, 10, 13]
category = [1, 3, 5, 6, 7, 8, 9, 13]
df_data = df.values
df_test_data = test_df.values

df_data[:, 10] = ((df_data[:, 10] != 0) + 0)
df_data[:, 11] = ((df_data[:, 11] != 0) + 0)

# df_data[:, 0] = (df_data[:, 0]/10).astype(int)
# df_test_data[:, 0] = (df_test_data[:, 0]/10).astype(int)

# df_test_data[:, 10] = ((df_test_data[:, 10] != 0) + 0)
# df_test_data[:, 11] = ((df_test_data[:, 11] != 0) + 0)

def Normalize(data1, data2, mode):
    if mode == 2 :
        for i in range(feature.shape[0] - 1):
            if i not in category:
                mean = np.mean(np.concatenate((data1[:, i], data2[:, i])))
                std = np.std(np.concatenate((data1[:, i], data2[:, i])))
                data1[:, i] = (data1[:, i] - mean) / (std + 1e-8)
                data2[:, i] = (data2[:, i] - mean) / (std + 1e-8)
            else:
                pass
        
        return data1, data2
    else:
        for i in range(feature.shape[0] - 1):
            if i not in category:
                mean = np.mean(data1[:, i])
                std = np.std(data1[:, i])
                data1[:, i] = (data1[:, i] - mean) / (std + 1e-8)
            else:
                pass
        
        return data1
    
feature_engeering = False
if feature_engeering:
    data = df_data
    test = df_test_data
else:
    # data, test = Normalize(df_data, df_test_data, 2)
    data = Normalize(df_data, df_data, 1)
    test = Normalize(df_test_data, df_test_data, 1)

# the number of some categpry and what category there are
dict_number = {feature[i] : len(set(data[:, i])) for i in range(data.shape[1])}
dict_category = {feature[i] : list(set(data[:, i])) for i in range(data.shape[1])}
# the number of some categpry and what category there are
test_number = {feature[i] : len(set(test[:, i])) for i in range(test.shape[1])}
test_category = {feature[i] : list(set(test[:, i])) for i in range(test.shape[1])}

# pre one-hot-encoding
dict_one_hot = {cate:{} for cate in category}
for k in dict_one_hot.keys():
    for s in dict_category[feature[k]]:
        zero = np.zeros((len(dict_category[feature[k]])))
        zero[dict_category[feature[k]].index(s)] = 1
        dict_one_hot[k].setdefault(s, zero.reshape(1, -1))

# total the number of columns
num_columns = 0
for i in range(feature.shape[0] - 1):
    if i in category:
        num_columns += list(dict_number.values())[i]
    else:
        num_columns += 1

# data by one-hot-encodeing
def one_hot_data(data, mode):
    raw_data = np.empty((data.shape[0], num_columns))
    shape =  data.shape[1] - 1 if mode else data.shape[1]
    for j in range(raw_data.shape[0]):
        a = np.array(data[j, 0]).reshape((1, 1))
        for i in range(1, shape): 
            b = dict_one_hot[i][data[j, i]] if i in category else np.array(data[j, i]).reshape((1, 1))
            a = np.concatenate((a, b), axis=1)
        raw_data[j] = a

    if mode:
        raw_label = np.empty((data.shape[0], 1))
        for j in range(raw_data.shape[0]):
            raw_label[j] = 0 if data[j, -1] == '<=50K' else 1   
        return raw_data, raw_label
    else:
        return raw_data
    
raw_data, raw_label = one_hot_data(data, True)
raw_test = one_hot_data(test, False)

def sigmoid(z):    
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def derivate_sigmoid(z):
        return np.multiply(z, 1.0 - z)
def split_data(rate, suffle):
      train_shape = int(rate * raw_data.shape[0])
      train_x = raw_data[:train_shape]
      train_y = raw_label[:train_shape]
      test_x = raw_data[train_shape:]
      test_y = raw_label[train_shape:]
      return train_x, train_y, test_x, test_y
def suffle(train_x, train_y):
      random_ = np.arange(train_x.shape[0])
      np.random.shuffle(random_)
      return train_x[random_], train_y[random_]
def _h(x, w, b):
      return sigmoid(np.dot(x, w) + b)
def _entropy(y_train, y_hat):
      epsilon = 1e-5
      entropy_loss = ((-y_train.T @ np.log(y_hat + epsilon)) - ((1 - y_train).T @ np.log(1 - y_hat + epsilon))) / len(y_train)
      return entropy_loss.item()
def save(result):
      result = np.round(result)
      final = np.empty((len(result), 2)).astype(object)
      for i in range(len(final)):
            final[i, 0] = str(i)
            final[i, 1] = str(int(result[i][0]))

      result_df = pd.DataFrame(final, columns = ['id','income'])
      result_df.to_csv('STUDENT_ID.csv', index=None)

train_x, train_y, test_x, test_y = split_data(0.5, True)

w = np.zeros((train_x.shape[1], 1))
b = np.zeros((1))
gradient = np.zeros((train_x.shape[1], 1))

iteration_time = 30
learning_rate = 0.2
batch_size = 64

train_entropy_loss = []
train_acc = []
test_entropy_loss = []
test_acc = []
for t in tqdm.tqdm(range(iteration_time)):

    x_train, y_train = suffle(train_x, train_y)

    for indx in range(int(np.floor(x_train.shape[0] / batch_size))):
        X = x_train[indx * batch_size : (indx + 1) * batch_size]
        Y = y_train[indx * batch_size : (indx + 1) * batch_size]
        y_pred = _h(X, w, b)
        error = Y - y_pred
        w_grad = -np.sum(np.dot(X.T, error), 1).reshape(-1, 1)
        b_grad = -np.sum(error).reshape((1))
        w = w - learning_rate/np.sqrt(t + 1) * w_grad
        b = b - learning_rate/np.sqrt(t + 1) * b_grad

    Y_hat = _h(x_train, w, b)
    train_entropy_loss.append(_entropy(y_train, Y_hat))
    train_acc.append(sum(np.round(Y_hat) == train_y) / len(train_y))

    
    test_y_hat = _h(test_x, w, b)
    test_entropy_loss.append(_entropy(test_y, test_y_hat))
    test_acc.append(sum(np.round(test_y_hat) == test_y) / len(test_x))

print(f'The last train accuracy: {train_acc[-1].item()}')
print(f'The last test accuracy: {test_acc[-1].item()}')
print(f'The percent of ground truth train label: {sum(np.round(train_y))/len(train_y)}')
print(f'The percent of ground truth test label: {sum(np.round(test_y))/len(test_y)}')
print(f'The percent of prediction test label: {sum(np.round(test_y_hat))/len(test_y_hat)}')


def relationship(j, mode):
    selection = feature[j]
    
    if mode == 'train':
        d = {dict_category[selection][i] : [0, 0] for i in range(len(dict_category[selection]))}
        for i in range(data.shape[0]):
            if data[i, -1] == dict_category[feature[-1]][0]:
                d[data[i, j]][0] = d[data[i, j]][0] + 1
            elif data[i, -1] == dict_category[feature[-1]][1]:
                d[data[i, j]][1] = d[data[i, j]][1] + 1
            else:
                raise
        x0 = np.arange(len(d))
        x1 = x0 + 0.25
        y0 = [list(d.values())[i][0] / sum(list(d.values())[i]) for i in range(len(d))]
        y1 = [list(d.values())[i][1] / sum(list(d.values())[i]) for i in range(len(d))]
        plt.bar(x0, y0, width=0.25, label='<=50K')
        plt.bar(x1, y1, width=0.25, label='>50K')
        if selection != feature[category[-1]]:
            plt.xticks(x0)
        else:
            pass
        plt.title(f'{feature[j]}')
        plt.legend()
        plt.show()

    elif mode == 'test':
        d1 = {dict_category[selection][i] : [0] for i in range(len(dict_category[selection]))}
        d2 = {test_category[selection][i] : [0] for i in range(len(test_category[selection]))}
        for i in range(data.shape[0]):
            d1[data[i, j]][0] = d1[data[i, j]][0] + 1
        for i in range(test.shape[0]):
            d2[test[i, j]][0] = d2[test[i, j]][0] + 1
        x1 = np.arange(len(d1))
        x2 = np.arange(len(d2))
        plt.plot(x1, list(d1.values()), label='train')
        plt.plot(x2, list(d2.values()), label='test')
        plt.title(feature[j])
        plt.legend()
        plt.show()
        
    else:
        raise


# for i in category:
    # relationship(i, 'train')