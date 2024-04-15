import numpy as np
from PIL import Image
import os, re
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def readPGM(filename, image_size):
    image = Image.open(filename).resize(image_size)
    image = np.array(image)
    label = int(re.findall(r'subject(\d+)', filename)[0])
    return image.astype(np.float64), label

def readData(dir_path, image_size, mode):
    file_list = os.listdir(f'{dir_path}/{mode}')
    num_files = len(file_list)
    D = np.empty((num_files, image_size[1], image_size[0])).astype(float)
    L = np.zeros(num_files).astype(int)
    
    for idx, file_name in enumerate(file_list):
        file_path = f'{dir_path}/{mode}/{file_name}'
        D[idx], L[idx] = readPGM(file_path, image_size)
        
    D = D.reshape(num_files, -1)
    return D, L

def Get_eigen(matrix, dims):
    
    try:
        eigen_val, eigen_vec = np.linalg.eigh(matrix)
    except:
        eigen_val, eigen_vec = np.linalg.eig(matrix)
        pass
    
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.linalg.norm(eigen_vec[:, i])
        
    idx = np.argsort(eigen_val)[::-1]
    W = eigen_vec[:, idx][:, :dims].real
    
    return W

def PCA(X, dims):
    mu = np.mean(X, axis=0)
    cov = (X - mu).T @ (X - mu)
    W = Get_eigen(cov, dims)
    return W

def kernel_pca(X, total_size, dims, kernel_type, gamma):
    kernel = get_kernel(X, kernel_type, gamma)
    matrix_n = np.ones((total_size, total_size)).astype(float) / total_size
    matrix = kernel - matrix_n @ kernel - kernel @ matrix_n + matrix_n @ kernel @ matrix_n
    W = Get_eigen(matrix, dims)
    return W

def LDA(X, Y, total_size, dims):
    classes, num_each_class = np.unique(Y, return_counts=True)
    num_class = len(num_each_class)

    mean_X = np.mean(X, axis=0)
    mean_class = np.zeros((num_class, total_size))

    for idx, label in enumerate(classes):
        mean_class[idx] = np.mean(X[Y == label], axis=0)
    
    cov_B = np.zeros((total_size, total_size))
    for idx, nums in enumerate(num_each_class):
        difference = (mean_class[idx] - mean_X).reshape((total_size, 1))
        cov_B += nums * difference @ difference.T
    
    cov_W = np.zeros((total_size, total_size))
    for idx, label in enumerate(classes):
        difference = np.array(X[Y == label] - mean_class[idx])
        cov_W += difference.T @ difference
        
    S_W_inv_S_B = np.linalg.pinv(cov_W) @ cov_B

    W = Get_eigen(S_W_inv_S_B, dims)
        
    return W

def get_kernel(data, kernel_type, gamma):
    if kernel_type == 'linear':
        return data.T @ data
    elif kernel_type == 'rbf':
        return np.exp(-gamma * cdist(data.T, data.T, 'sqeuclidean'))
    elif kernel_type == 'poly':
        return (1 + data.T @ data) ** 2
    else:
        raise ValueError
    
def kernel_lda(X, Y, total_size, dims, kernel_type, gamma):
    
    classes, num_each_class = np.unique(Y, return_counts=True)
    num_class = len(num_each_class)
    kernel_class = np.zeros((num_class, total_size, total_size))
    for idx, label in enumerate(classes):
        images = X[Y == label]
        kernel_class[idx] = get_kernel(images, kernel_type, gamma)
    kernel_all = get_kernel(X, kernel_type, gamma)
        
    matrix_n = np.zeros((total_size, total_size))
    identity_matrix = np.eye(total_size)
    ones_matrix = np.ones((total_size, total_size))
    for idx, num in enumerate(num_each_class):
        matrix_n += kernel_class[idx] @ (identity_matrix -  ones_matrix / num) @ kernel_class[idx].T

    matrix_m_i = np.zeros((num_class, total_size))
    for idx, kernel in enumerate(kernel_class):
        for row_idx, row in enumerate(kernel):
            matrix_m_i[idx, row_idx] = np.sum(row) / num_each_class[idx]
            
    matrix_m_star = np.zeros(total_size)
    for idx, row in enumerate(kernel_all):
        matrix_m_star[idx] = np.sum(row) / len(X)
        
    matrix_m = np.zeros((total_size, total_size))
    for idx, num in enumerate(num_each_class):
        difference = (matrix_m_i[idx] - matrix_m_star).reshape(total_size, 1)
        matrix_m += num * difference @ difference.T

    matrix = np.linalg.pinv(matrix_n) @ matrix_m
    W = Get_eigen(matrix, dims)
    
    return W
    

def Projection(D, W):
    return D @ W

def error_rate(X, Y, test_X, test_Y, W, mode, kernel_type):
    projection_X = Projection(X, W)
    projection_test_X = Projection(test_X, W)
    error = 0
    distance = np.zeros(len(X))
    for test_idx, test in enumerate(projection_test_X):
        for train_idx, train in enumerate(projection_X):
            distance[train_idx] = np.linalg.norm(test - train)
        min_distances = np.argsort(distance)[:5]
        predict = np.argmax(np.bincount(Y[min_distances]))
        if predict != test_Y[test_idx]:
            error += 1
    print(f'times: {error} | rate: {error / len(test_X):.3f}')
    
    with open('record.txt', 'a') as f:
        if kernel_type == '':
            f.write(f'{mode} | times: {error} | rate: {error / len(test_X):.3f} \n')
        else:
            f.write(f'{mode} | {kernel_type} | times: {error} | rate: {error / len(test_X):.3f} \n')
        f.close()
    
    
def eigenface(W, image_size, save_file, mode, kernel_type):
    plt.clf()
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(5, 5, idx + 1)
            plt.imshow(W[:, idx].reshape(image_size[::-1]), cmap='gray')
            plt.axis('off')

    plt.savefig(f'./{save_file}/eigenface/{mode}_{kernel_type}.png')

def reconstruction(X, W, image_size, save_file, mode, kernel_type):
    plt.clf()
    choice = np.random.choice(len(X), 10)
    choice_X = X[choice]
    reconstruct_X = choice_X @ W @ W.T
    for idx in range(10):
        plt.subplot(10, 2, idx * 2 + 1)
        plt.imshow(choice_X[idx].reshape(image_size[::-1]), cmap='gray')
        plt.axis('off')
        
        plt.subplot(10, 2, idx * 2 + 2)
        plt.imshow(reconstruct_X[idx].reshape(image_size[::-1]), cmap='gray')
        plt.axis('off')

    plt.savefig(f'./{save_file}/reconstruction/{mode}_{kernel_type}.png')

if __name__ == '__main__':
    
    '''
        Run all experiments: python3 eigenface.py
        Run one: python3 eigenface.py ['pca', 'lda', 'kernel_pca', 'kernel_lda'] ['linear', 'rbf', 'poly']
    '''
    
    image_size = (19, 23)
    # image_size = (38, 46)
    # image_size = (57, 69)
    # image_size = (195, 231)
    total_size = image_size[0] * image_size[1]
    dir_path = './Yale_Face_Database'
    first_k = 25
    k_neighbor = 5
    gamma = 1e-6
    
    save_file = './save'
    os.makedirs(f'{save_file}/eigenface', exist_ok=True)
    os.makedirs(f'{save_file}/reconstruction', exist_ok=True)
    
    X, Y = readData(dir_path, image_size, 'Training')
    test_X, test_Y = readData(dir_path, image_size, 'Testing')

    if len(sys.argv) == 3:
        
        mode = sys.argv[1]
        kernel_type = sys.argv[2]
        
        if mode == 'pca':
            W = PCA(X, first_k)
        elif mode == 'lda':
            W = LDA(X, Y, total_size, first_k)
        elif mode == 'kernel_pca':
            W = kernel_pca(X, total_size, first_k, kernel_type, gamma)
        elif mode == 'kernel_lda':
            W = kernel_lda(X, Y, total_size, first_k, kernel_type, gamma)
        else:
            raise ValueError
        
        kernel_type = ''
        
        error_rate(X, Y, test_X, test_Y, W, mode, kernel_type)
        eigenface(W, image_size, save_file, mode, kernel_type)
        reconstruction(X, W, image_size, save_file, mode, kernel_type)
        
    else:
        modes = ['pca', 'lda', 'kernel_pca', 'kernel_lda']
        kernel_types = ['linear', 'rbf', 'poly']
        
        for mode in modes:
            if mode in ['pca', 'lda']:
                if mode == 'pca':
                    W = PCA(X, first_k)
                elif mode == 'lda':
                    W = LDA(X, Y, total_size, first_k)
                elif mode == 'kernel_pca':
                    W = kernel_pca(X, total_size, first_k, kernel_type, gamma)
                elif mode == 'kernel_lda':
                    W = kernel_lda(X, Y, total_size, first_k, kernel_type, gamma)
                else:
                    raise ValueError
                
                kernel_type = ''
                
                error_rate(X, Y, test_X, test_Y, W, mode, kernel_type)
                eigenface(W, image_size, save_file, mode, kernel_type)
                reconstruction(X, W, image_size, save_file, mode, kernel_type)
            
            else: 
                for kernel_type in kernel_types:
                    if mode == 'pca':
                        W = PCA(X, first_k)
                    elif mode == 'lda':
                        W = LDA(X, Y, total_size, first_k)
                    elif mode == 'kernel_pca':
                        W = kernel_pca(X, total_size, first_k, kernel_type, gamma)
                    elif mode == 'kernel_lda':
                        W = kernel_lda(X, Y, total_size, first_k, kernel_type, gamma)
                    else:
                        raise ValueError

                    error_rate(X, Y, test_X, test_Y, W, mode, kernel_type)
                    eigenface(W, image_size, save_file, mode, kernel_type)
                    reconstruction(X, W, image_size, save_file, mode, kernel_type)