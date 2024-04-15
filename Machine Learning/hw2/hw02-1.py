import gzip
import numpy as np
import matplotlib.pyplot as plt
import tqdm

mode = input("Input 0 or 1 (0: discrete mode; 1: continuous mode): ")

train_images_file_path = 'train-images-idx3-ubyte.gz'
train_labels_file_path = 'train-labels-idx1-ubyte.gz'
test_images_file_path = 't10k-images-idx3-ubyte.gz'
test_labels_file_path = 't10k-labels-idx1-ubyte.gz'

bin_size = 8
image_size = 28 * 28
image_offset = 16
label_offset = 8

def get(image_file_path, label_file_path):
    
    file = gzip.open(image_file_path, 'rb')
    file_read = file.read(image_offset)
    
    buff = file.read()
    
    image_data = np.frombuffer(buff, dtype=np.uint8)
    image_data = np.reshape(image_data, (-1, 28, 28)).astype(int)
    
    file.close()
    
    print(f"Found {(len(buff) / image_size) :.0f} images")
    
    file = gzip.open(label_file_path, 'rb')
    file_read = file.read(label_offset)
    
    buff = file.read()
    
    label_data = np.frombuffer(buff, dtype=np.uint8)
    label_data = np.array(label_data).astype(int)
    
    file.close() 
    
    print(f"Found {(len(buff)) :.0f} labels")
    
    del file_read
    del buff
    
    image_data = image_data / bin_size
    image_data = image_data.astype(int)
    
    return image_data, label_data

X, Y = get(train_images_file_path, train_labels_file_path)
test_X, test_Y = get(test_images_file_path, test_labels_file_path)

n_X = len(X)
n_test_X = len(test_X)
class_number = len(set(test_Y))

if mode == "1":
    X = np.reshape(X, (n_X, image_size))
    test_X = np.reshape(test_X, (n_test_X, image_size, 1))
elif mode == "0":
    pass
else:
    raise KeyError

class_count = [np.sum(np.where(Y == c, 1, 0)).astype(int) for c in range(class_number)]
class_prob = np.array([class_count[c] / n_X for c in range(class_number)])

if mode == "1":
    cov_bin_x = np.empty((class_number, image_size, image_size))
    mu_bin_x = np.empty((class_number, image_size, 1))


    for c in tqdm.tqdm(range(class_number)):

        sub_bin_x = X[np.where(Y == c)]
        
        mu_bin_x[c] = np.expand_dims(np.mean(sub_bin_x, axis=0), axis=1)
        
        cov_bin_x[c] = np.cov(sub_bin_x, rowvar=0, bias=1) + np.eye(image_size) * 2

    for i in range(class_number):
        cov_bin_x[i] *= (class_count[i] - 1)
        cov_bin_x[i] /= class_count[i]

    inv_cov = np.array([np.linalg.inv(cov_bin_x[c]) for c in range(class_number)])
    # det_cov = np.array([np.linalg.det(cov_bin_x[c]) for c in range(class_number)])
    # log_det_cov = np.log2(det_cov)
    
    def log_gaussian(mu, inv_sigma, X):
        return - (((X - mu).T @ inv_sigma @ (X - mu)))
    
else:
    ## 統計各類的 class 中 每個 pixel 中 0~31 bin 的機率
    bin_prob = np.zeros((10, 32, 28, 28))
    for i in tqdm.tqdm(range(class_number)):
        for j in range(32):
            x_j = np.where(X[[np.where(Y == i)[0]]] == j, 1, 0)[0]
            
            bin_prob[i, j] = np.sum(x_j, axis=0)
            
    for j in range(class_number):
        bin_prob[j] = bin_prob[j] / class_count[j]
        
    ## 將機率是 0 的 改成 32個之中 除了0之外 最小的
    for i in tqdm.tqdm(range(class_number)):
        for j in range(28):
            for k in range(28):
                bin_prob[i, :, j, k][np.where(bin_prob[i, :, j, k] == 0)] = min(bin_prob[i, :, j, k][np.where(bin_prob[i, :, j, k] != 0)]) / 10

    log_prob = np.log2(bin_prob)

error = 0
test_prob = np.empty((n_test_X, class_number))
test_pred = np.zeros((n_test_X)).astype(int)
if mode == "1":
    for ind in tqdm.tqdm(range(n_test_X)):

        naive_list = []
        for c in range(class_number):
        
            naive_list.append(log_gaussian(mu_bin_x[c], inv_cov[c], test_X[ind])[0, 0])
            # naive_list.append(log_gaussian(mu_bin_x[i], inv_cov[i], test_X[ind]).item() - log_det_cov[i])
        
        naive_list /= sum(naive_list)
        test_prob[ind] = naive_list
        
        predtion = np.where(naive_list == min(naive_list))[0].item()
        test_pred[ind] = predtion
        
        if predtion != test_Y[ind]:
            error += 1
    
else:
    for ind in tqdm.tqdm(range(n_test_X)):

        naive_list = []
        for c in range(class_number):

            pr = np.log2(class_prob[c])
            
            for i in range(28):
                for j in range(28):
                    pr += log_prob[c, test_X[ind, i, j], i, j]
            
            naive_list.append(pr)
            
        naive_list /= sum(naive_list)
        test_prob[ind] = naive_list
        
        predtion = np.where(naive_list == min(naive_list))[0].item()
        test_pred[ind] = predtion
        
        if predtion != test_Y[ind]:
            error += 1

for id in range(n_test_X):
    print("Postirior (in log scale):")
    for c in range(class_number):
        print(f"{c}: {test_prob[id][c]}")

    print(f"Prediction: {test_pred[id].item()}, Ans: {test_Y[id]}\n")
    
classifier = np.empty((class_number, 28, 28)).astype(str)

if mode == "1":
    mu_bin_x_image = mu_bin_x.reshape(class_number, 28, 28)
    for i in range(class_number):
        for j in range(28):
            for k in range(28):
                classifier[i, j, k] = '0' if mu_bin_x_image[i, j, k] <= 16 else "1"
else:    
    for i in range(class_number):
        for j in range(28):
            for k in range(28):
                classifier[i, j, k] = '0' if sum(bin_prob[i, :16, j, k]) > sum(bin_prob[i, 16:, j, k]) else '1'

for digit in range(class_number):
    for i in range(28):
        print(" ".join(classifier[digit][i].tolist()))


print("\n")
print(f'Error rate: {error / n_test_X}')