from scipy.spatial.distance import cdist
import numpy as np
from PIL import Image
import random
import sys
import matplotlib.pyplot as plt
import os

def load_image(image_name):
    img = Image.open(image_name)
    img = np.array(img.getdata()) #(10000,3)
    return img

def RBF_kernel(X, gamma_s, gamma_c):
    color_distance = cdist(X, X,'sqeuclidean') #(10000,10000)

    grid = np.indices((100, 100)).reshape(2, -1).T
    spatial_distance = cdist(grid, grid, 'sqeuclidean')
    kernel = np.exp(-gamma_s * spatial_distance) * np.exp(-gamma_c * color_distance)
    
    return kernel

def initial_center(cluster_number, mode, grid):
    if mode=="random":
        centers_idx = list(random.sample(range(0,10000), cluster_number))
        print(f"Find the all initial centers")
    elif mode=="kmeans++":
        centers_idx = []
        centers_idx = list(random.sample(range(0,10000), 1))
        found = 1
        print(f"Find the {found}-th initial center")
        while (found < cluster_number):
            dist = np.zeros(10000)
            for i in range(10000):
                min_dist = np.Inf
                for f in range(found):
                    tmp = np.linalg.norm(grid[i, :] - grid[centers_idx[f], :])
                    if tmp < min_dist:
                        min_dist = tmp
                dist[i] = min_dist
            dist = dist / np.sum(dist)
            idx = np.random.choice(np.arange(10000), 1, p=dist)
            centers_idx.append(idx[0])
            found += 1
            print(f"Find the {found}-th initial center")

    return centers_idx

def initial_kernel_kmeans(cluster_number, kernel, mode):
      
    cluster = []
    for i in range(cluster_number):
        cluster.append(kernel[i, :])
    return cluster


def construct_Laplacian(kernel):
    d = np.sum(kernel, axis=1)
    D = np.diag(d)
    L = np.array(D - kernel)
    
    return D, L

def clustering(cluster_number, kernel, cluster):
    N = len(kernel)
    new_cluster = np.zeros(N, dtype=int)
    for n in range(N):
        c = -1
        min_dist = np.Inf
        for k in range(cluster_number):
            dist = np.linalg.norm((kernel[n] - kernel[k]), ord=2)
            if dist < min_dist:
                c = k
                min_dist = dist
        new_cluster[n] = c
    return new_cluster

def D_square_root(D):
    Dsym = np.array(np.diag(np.power(np.diag(D), -0.5)))
    return Dsym

def normalize(U, cluster_number):
    sum_U = np.sum(U, axis=1).reshape(-1, 1)
    sigma = np.hstack((sum_U, sum_U))
    for i in range(cluster_number - 2):
        sigma = np.hstack((sigma, sum_U))
    T = U / sigma
    return T


def save_png(N, cluster, iter):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], 
                       [0, 255, 255], [255, 0, 255], [255, 255, 0],
                       [255, 255, 255]])
    
    if cluster_number > 7:
        colors = np.append(colors, np.random.choice(256, (cluster_number - 7, 3)), axis=0)
        
    result = np.zeros((100 * 100, 3))
    for n in range(N):
        result[n, :] = colors[cluster[n], :]
    
    img = result.reshape(100, 100, 3)
    img = Image.fromarray(np.uint8(img))
    
    return img

def kernel_kmeans(cluster_number, kernel, cluster, iteration, filename, result):
    
    # img = [save_png(len(kernel), cluster, 0)]
    img = []
    runs = 1
    while True:
        print(f"----- Iteration {runs: 3d} -----")
        new_cluster = clustering(cluster_number, kernel, cluster)

        if runs != 1:
            if(np.linalg.norm((new_cluster - cluster), ord=2) < 1e-4) or iteration <= runs:
                break
        
        cluster = new_cluster
        img.append(save_png(len(kernel), cluster, runs))
        runs += 1
    
    if result:
        colors = ['r' if c == 0 else 'b' for c in cluster]
        plt.scatter(kernel[:, 0], kernel[:, 1], c=colors)
        plt.show()
    else:
        print()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=200)
        return None

if __name__ == '__main__':
    
    output_dir = 'Spectral-Clustering'
    if len(sys.argv) < 5:
        print('Please choose image_number(1~2), cluster_number, mode(kmeans++ or random), and cut(ratio or normalize)')
        print('For example, python3 xxx.py 2 2 kmeans++ ratio')
        exit()
    ind = int(sys.argv[1])
    cluster_number = int(sys.argv[2])
    mode = sys.argv[3]      # kmeans++ or random
    try:
        result = sys.argv[4]
    except:
        result = False
    
    if mode != 'kmeans++' and mode != 'random':
        raise KeyError
    else:
        pass
    
    cut = sys.argv[4]
    image_name = f'image{ind}.png'

    gamma_s = 1e-4
    gamma_c = 1e-3
    iteration = 200
    os.makedirs(output_dir, exist_ok=True)
    test = f'test_{ind}_{cluster_number}_{cut}'
    os.makedirs(output_dir + '/' + test, exist_ok=True)
    filename = f'./{output_dir}/{mode}/{cut}/Image_{ind}_Cluester_{cluster_number}.gif'
    
    if cut == 'ratio':
        if result:
            print("Loading weight")
            eigenvalue = np.load(output_dir + '/' + test + "/eigenvalue.npy")
            eigenvector = np.load(output_dir + '/' + test + "/eigenvector.npy")
            eigenvector = eigenvector.T
            D = np.load(output_dir+ '/' + test + "/D.npy")
            L = np.load(output_dir + '/' + test + "/L.npy")
            print("Success Loading")
            
        else:
            X_color = load_image(image_name) #(10000, 3)
            kernel = RBF_kernel(X_color, gamma_s, gamma_c) #spatial, color
            D, L = construct_Laplacian(kernel)
            
            np.save(output_dir + '/' + test + "/D.npy", D)
            np.save(output_dir + '/' + test + "/L.npy", L)
            print("Calulating eigenvalues and eigenvectors !")
            
            eigenvalue, eigenvector = np.linalg.eig(L)
            eigenvector = eigenvector.T
            
            np.save(output_dir + '/' + test + "/eigenvalue.npy", eigenvalue)
            np.save(output_dir + '/' + test + "/eigenvector.npy", eigenvector)
            print("Success calculating")
        
        
        
        
        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx] > 0
        sort_idx = sort_idx[mask]
        U = eigenvector[sort_idx[:cluster_number]].T
        
        cluster = initial_kernel_kmeans(cluster_number, U, mode)
        kernel_kmeans(cluster_number, U, cluster, iteration, filename, result)
    
    elif cut == 'normalize':
        if result:
            print("Loading weight")
            eigenvalue = np.load(output_dir + '/' + test + "/eigenvalue.npy")
            eigenvector = np.load(output_dir + '/' + test + "/eigenvector.npy")
            eigenvector = eigenvector.T
            D_sym = np.load(output_dir+ '/' + test + "/D_sym.npy")
            L_sym = np.load(output_dir + '/' + test + "/L_sym.npy")
            print("Success Loading")
            
        else:
            X_color = load_image(image_name) #(10000, 3)
            kernel = RBF_kernel(X_color, gamma_s, gamma_c) #spatial, color
            D, L = construct_Laplacian(kernel)
            
            D_sym = D_square_root(D)
            L_sym = D_sym @ L @ D_sym
            np.save(output_dir + '/' + test + "/D_sym.npy", L_sym)
            np.save(output_dir + '/' + test + "/L_sym.npy", L_sym)
            
            print("Calulating eigenvalues and eigenvectors !")
            eigenvalue, eigenvector = np.linalg.eig(L_sym)
            eigenvector = eigenvector.T
            np.save(output_dir + '/' + test + "/eigenvalue.npy", eigenvalue)
            np.save(output_dir + '/' + test + "/eigenvector.npy", eigenvector)
            print("Success calculating")
        
        
        sort_idx = np.argsort(eigenvalue)
        mask = eigenvalue[sort_idx] > 0
        sort_idx = sort_idx[mask]
        
        U = eigenvector[sort_idx[:cluster_number]].T
        T = normalize(U, cluster_number)
        
        cluster = initial_kernel_kmeans(cluster_number, T, mode)
        kernel_kmeans(cluster_number, T, cluster, iteration, filename, result)
    else:
        raise ValueError