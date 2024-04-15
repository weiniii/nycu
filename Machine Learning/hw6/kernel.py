from scipy.spatial.distance import pdist, cdist
import numpy as np
from PIL import Image
import random
import sys

import os

def load_image(image_name):
    img = Image.open(image_name)
    img = np.array(img.getdata()) #(10000,3)
    return img

def RBF_kernel(X, gamma_s, gamma_c):
    color_distance = cdist(X, X,'sqeuclidean')
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
    else:
        raise KeyError
    return centers_idx

def initial_kernel_kmeans(cluster_number, kernel, mode):
    grid = np.indices((100, 100)).reshape(2, -1).T
    if mode == "kmeans++":
        centers_idx = initial_center(cluster_number, mode, grid)
    elif mode == "random":
        centers_idx = initial_center(cluster_number, mode, None)
        
    N = len(kernel)
    cluster = np.zeros(N, dtype=int)
    for n in range(N):
        dist = np.zeros(cluster_number)
        for k in range(cluster_number):
            dist[k] = kernel[n, n] + kernel[centers_idx[k],centers_idx[k]] - 2 * kernel[n,centers_idx[k]]
        cluster[n] = np.argmin(dist)
    return cluster

def construct_sigma_n(kernelj, cluster, cluster_k):
    ker = kernelj.copy()
    mask = np.where(cluster == cluster_k)
    sigma = np.sum(ker[mask])
    return sigma

def construct_sigma_pq(C, cluster_number, kernel, cluster):
    pq = np.zeros(cluster_number)
    for k in range(cluster_number):
        ker = kernel.copy()
        for n in range(len(kernel)):
            if cluster[n] != k:
                ker[n, :] = 0
                ker[:, n] = 0
        pq[k] = np.sum(ker) / C[k] / C[k]
    return pq

def construct_C(cluster_number, cluster):
    C = np.zeros(cluster_number, dtype=int)
    for k in range(cluster_number):
        indicator = np.where(cluster == k, 1, 0)
        C[k] = np.sum(indicator)
    return C

def clustering(cluster_number, kernel, cluster):
    N = len(kernel)
    new_cluster = np.zeros(N, dtype=int)
    C = construct_C(cluster_number, cluster)
    pq = construct_sigma_pq(C, cluster_number, kernel, cluster)
    for j in range(N):
        dist = np.zeros(cluster_number)
        for k in range(cluster_number):
            dist[k] += kernel[j, j] + pq[k]
            dist[k] -= 2 / C[k] * construct_sigma_n(kernel[j, :], cluster, k)
        new_cluster[j] = np.argmin(dist)

    return new_cluster

def save_image(N, cluster, iter):
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

def kernel_kmeans(cluster_number, kernel, cluster, iteration):
    img = [save_image(len(kernel), cluster, 0)]
    runs = 1
    while True:
        print(f"----- Iteration {runs: 3d} -----")
        new_cluster = clustering(cluster_number, kernel, cluster)

        if(np.linalg.norm((new_cluster - cluster), ord=2) < 1e-4) or iteration <= runs:
            break
        
        cluster = new_cluster
        img.append(save_image(len(kernel), cluster, runs))
        runs += 1
    
    print()
    filename = f'./{output_dir}/{mode}/Image_{ind}_Cluester_{cluster_number}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=200)

if __name__ == '__main__':
    
    output_dir = 'Kernel-Kmeans'
    if len(sys.argv) < 4:
        print('Please choose image_number(1~2), cluster_number, and mode(kmeans++ or random)')
        print('For example, python3 xxx.py 2 2 kmeans++')
        exit()
    ind = int(sys.argv[1])
    cluster_number = int(sys.argv[2])
    mode = sys.argv[3]      # kmeans++ or random
    image_name = f'image{ind}.png'

    gamma_s = 1e-4
    gamma_c = 1e-3
    iteration = 200
    os.makedirs(output_dir, exist_ok=True)
    
    X_color = load_image(image_name) 
    kernel = RBF_kernel(X_color, gamma_s, gamma_c)
    cluster = initial_kernel_kmeans(cluster_number, kernel, mode)
    kernel_kmeans(cluster_number, kernel, cluster, iteration)