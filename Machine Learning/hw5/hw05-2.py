import csv
import sys
import time
import numpy as np
from libsvm.svmutil import *

kernel = {
    'linear': 0, 
    'polynomial': 1, 
    'RBF': 2, 
}

def openCSV(filename):
    with open(filename, 'r') as file:
        content = list(csv.reader(file))
        content = np.array(content)
    return content

def train(Y, X, kernel):
    return svm_train(Y, X, f'-t {kernel} -q')

def compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt):
    print(opt)
    cv_acc = svm_train(Y, X, opt)
    if cv_acc > optimal_cv_acc:
        return cv_acc, opt
    else:
        return optimal_cv_acc, optimal_opt

def gridSearch(X, Y):
    optimal_opt = ''
    optimal_cv_acc = 0
    costs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    cnt = 0
    for k in kernel:
        for cost in costs:
            if k == 'linear':
                opt = f'-t {kernel[k]} -c {cost} -v 3 -q'
                cnt += 1
                print(f'Linear Kernel | Run {cnt}:')
                optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
                print('-' * 60)
            elif k == 'polynomial':
                for gamma in gammas:
                    for degree in range(2, 5):
                        for coef0 in range(0, 3):
                            opt = f'-t {kernel[k]} -c {cost} -g {gamma} -d {degree} -r {coef0} -v 3 -q'
                            cnt += 1
                            print(f'Polynomial Kernel | Run {cnt}:')
                            optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
                            print('-' * 60)
            elif k == 'RBF':
                for gamma in gammas:
                    opt = f'-t {kernel[k]} -c {cost} -g {gamma} -v 3 -q'
                    cnt += 1
                    print(f'RBF Kernel | Run {cnt}:')
                    optimal_cv_acc, optimal_opt = compareAccuracy(Y, X, opt, optimal_cv_acc, optimal_opt)
                    print('-' * 60)
    print(f'Total combinations: {cnt}')
    print(f'Optimal cross validation accuracy: {optimal_cv_acc}')
    print(f'Optimal option: {optimal_opt}')

def linearKernel(X1, X2):
    kernel = X1 @ X2.T
    return kernel
    
def RBFKernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    kernel = np.exp((-1 * gamma * dist))
    return kernel

if __name__ == '__main__':
    START = time.time()

    if len(sys.argv) < 2:
        print('Please choose a part to run')
        print('Part 1. compare three types of kernel')
        print('Part 2. grid search')
        print('Part 3. user-defined kernel')
        print('Usage: python3 hw5_Q2.py {number of part}')
        exit()

    X_train = openCSV('./data/X_train.csv').astype(float)
    Y_train = list(openCSV('./data/Y_train.csv').astype(int).ravel())
    X_test = openCSV('./data/X_test.csv').astype(float)
    Y_test = list(openCSV('./data/Y_test.csv').astype(int).ravel())
    N_train = len(X_train)
    N_test = len(X_test)
    
    print('-' * 60)
    print(f'Start Part {sys.argv[1]}')
    print('-' * 60)
    if sys.argv[1] == '1':
        # Part 1. compare three types of kernel
        for k in kernel:
            print(f'Kernel: {k}')
            parameter = f'-t {kernel[k]} -d 2 -q' if kernel[k] == 1 else f'-t {kernel[k]} -q'
            
            m = svm_train(Y_train, X_train, parameter)
            res = svm_predict(Y_test, X_test, m)
            
            if kernel[k] != 2:
                print(f'Spend {time.time() - START:11.3f} second(s)')
                print('-' * 60)
                START = time.time()
            
    elif sys.argv[1] == '2':
        # Part 2. grid search
        gridSearch(X_train, Y_train)

        opt = '-t 2 -c 10 -g 0.01 -q'
        m = svm_train(Y_train, X_train, opt)
        res = svm_predict(Y_test, X_test, m)
    elif sys.argv[1] == '3':
        # Part 3. user-defined kernel
        print('Kernel: user-defined')
        linear_kernel = linearKernel(X_train, X_train)
        RBF_kernel = RBFKernel(X_train, X_train, 1 / 784)
        
        linear_kernel_s = linearKernel(X_train, X_test).T
        RBF_kernel_s = RBFKernel(X_train, X_test, 1 / 784).T
        
        X_kernel = np.hstack((np.arange(1, N_train + 1).reshape((-1, 1)), linear_kernel + RBF_kernel))
        X_kernel_s = np.hstack((np.arange(1, N_test + 1).reshape((-1, 1)), linear_kernel_s + RBF_kernel_s))

        opt = '-t 4 -q'
        m = svm_train(Y_train, X_kernel, opt)
        svm_predict(Y_test, X_kernel_s, m)
    else:
        print('Wrong part number')
    
    print(f'Spend {time.time() - START:11.3f} second(s)')
    print('-' * 60)