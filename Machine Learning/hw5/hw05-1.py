import numpy as np
import matplotlib.pyplot as plt
import scipy

def getData(filename):
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.split(' ')
            X.append(x)
            Y.append(y)
        
        X = np.array(X).astype(float).reshape(-1, 1)
        Y = np.array(Y).astype(float).reshape(-1, 1)
    return X, Y

def RationalQuadraticKernel(X1, X2, sigma, Alpha, length_scale):
    L2_dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    kernel = (sigma ** 2) * ((1 + L2_dist / (2 * Alpha * (length_scale ** 2))) ** (-1 * Alpha))
    return kernel

def NegativeLogLikelihood(theta, X, Y, beta):
    theta = theta.ravel()
    kernel = RationalQuadraticKernel(X, X, theta[0], theta[1], theta[2])
    kernel += np.eye(len(X)) / beta
    nll = np.sum(np.log(np.diagonal(np.linalg.cholesky(kernel)))) / 2
    nll += 0.5 * Y.T @ np.linalg.inv(kernel) @ Y
    nll += 0.5 * len(X) * np.log(2 * np.pi)
    return nll

def GaussianProcess(X, Y, TEST_X, beta, sigma, Alpha, length_scale):
    kernel = RationalQuadraticKernel(X, X, sigma, Alpha, length_scale)
    Cov = kernel + np.eye(len(X)) / beta
    C_inv = np.linalg.inv(Cov)
    kernel_s = RationalQuadraticKernel(X, TEST_X, sigma, Alpha, length_scale)
    kernel_ss = RationalQuadraticKernel(TEST_X, TEST_X, sigma, Alpha, length_scale) 

    mu_star = kernel_s.T @ C_inv @ Y
    var_star = kernel_ss + np.eye(len(TEST_X)) / beta
    var_star -= kernel_s.T @ C_inv @ kernel_s
    
    plt.plot(TEST_X, mu_star, color='b')
    plt.scatter(X, Y, color='k', s=10)
    
    interval = 1.96 * np.sqrt(np.diagonal(var_star))
    TEST_X = TEST_X.ravel()
    mu_star = mu_star.ravel()

    plt.plot(TEST_X, mu_star + interval, color='r')
    plt.plot(TEST_X, mu_star - interval, color='r')
    plt.fill_between(TEST_X, mu_star + interval, mu_star - interval, color='r', alpha=0.1)

    plt.title(f'sigma: {sigma:.5f}, alpha: {Alpha:.5f}, length scale: {length_scale:.5f}')
    plt.xlim(-60, 60)
    plt.show()
    
if __name__ == '__main__':
    
    file = './data/input.data'
    X, Y = getData(file)

    TEST_X = np.linspace(-60.0, 60.0, 1000).reshape(-1, 1)
    beta = 5
    sigma = 1
    Alpha = 1
    length_scale = 1

    GaussianProcess(X, Y, TEST_X, beta, sigma, Alpha, length_scale)

    opt = scipy.optimize.minimize(NegativeLogLikelihood, [sigma, Alpha, length_scale], 
                    bounds=((1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5)), 
                    args=(X, Y, beta))
    sigma_opt = opt.x[0]
    alpha_opt = opt.x[1]
    length_scale_opt = opt.x[2]
    GaussianProcess(X, Y, TEST_X, beta, sigma_opt, alpha_opt, length_scale_opt)
    