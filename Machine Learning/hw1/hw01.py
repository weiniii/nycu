import numpy as np

# Setting Precision
np.set_printoptions(precision=10, suppress=True)

# Around function
def around(arr):
    return np.around(arr, decimals=10)

# Input base n and lambda
print("Input an integer n as the number of polynomial base: ")
n = int(input())
print("Input a lambda: ")
lamb = float(input())

# Read data and compute A, X, B, (X_GD, Y for gradient decsent metho)
A = []
B = []
X = []
N_sample = 0
LR = 0.01
ITERATION_ROUND = 1000

f = open('testfile.txt','r')
for r in f.readlines():
    r = r.strip('\n')
    x = list(map(float,r.split(',')))
    for j in range(n):
        A.append(x[0] ** j)
    B.append(x[1])
    X.append(x)
    N_sample += 1
f.close()

A = np.array(A)
A = A.reshape(N_sample, n)
B = np.array(B)
B = B.reshape(N_sample, 1)
X = np.array(X)
X = X.T
X_GD = X[0, :]
Y = B.reshape(-1)

# LU decomposition
def LU(arr):
    L = np.eye(n)
    U = np.array(arr)
    for j in range(0, n - 1):
        for i in range(j + 1, n):
            reset = U[i][j] / U[j][j]
            U[i] -= U[j] * reset
            L[i][j] = reset
    return L, U

# Gaussian_Jordan to compute inverse matrix
def Gaussian_Jordan(arr):
    new_L1 = np.eye(n)
    new_U1 = np.array(arr)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            reset = new_U1[i][j] / new_U1[j][j]
            new_U1[i] -= new_U1[j] * reset
            new_L1[i] -= new_L1[j] * reset
    for i in range(n):
        new_L1[i] /= arr[i][i]
    return new_L1

# Matrix dot product
def dot(arr1, arr2):
    R = np.zeros((arr1.shape[0], arr2.shape[1]))
    if arr1.shape[1] != arr2.shape[0]:
        raise ValueError('Input error of array size !')
    for i in range(arr1.shape[0]):
        for j in range(arr2.shape[1]):
            r = 0
            for k in range(arr1.shape[1]):
                r += arr1[i, k] * arr2[k, j]
            R[i, j] = r
    return R

# Det of matrix
def detf(arr):
    
    if len(arr) == 1:
        return arr
    
    detval = 0
    for i in range(len(arr)):
        A1 = np.delete(arr, 0, 0)
        A1 = np.delete(A1, i, 1)
        detval += arr[0, i] * detf(A1) * ((-1) ** i)
    
    return detval

# Matrix inverse
def inverse(arr):
    n = len(arr)
    inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A1 = np.delete(arr, i, 0)
            A1 = np.delete(A1, j, 1)
            
            inv[i,j] = detf(A1) * (-1) ** (i + j)
    return inv.T / detf(arr)

# Show polynomial function
def showPolynomial(arr, n):
    s = ''
    for i in range(n):
        if i == 0:
            s += (str(arr[n - 1 - i]) + 'X^'+str(n - 1 - i))
        else:
            if arr[n - 1 - i] < 0:
                s += (' - ' + str(-arr[n - 1 - i]))
                if i != n-1:
                    s+=('X^'+str(n - 1 - i))
            else:
                s += (' + ' + str(arr[n - 1 - i]))
                if i != n-1:
                    s += ('X^' + str(n - 1 - i))
    print("Fitting Line: " + s)

# Show the image of function
def showResult(arr1, arr2, arr3, arr4):
    import matplotlib.pyplot as plt
    
    x1 = np.linspace(-6, 6, num=100)
    fx1 = []
    for i in range(len(x1)):
        fxx = 0
        for k in range(n):
            fxx += arr2[k] * (x1[i] ** k)
        fx1.append(fxx)
    
    plt.subplot(3, 1, 1)
    plt.scatter(arr1[0], arr1[1], c ="red", edgecolor = 'black')
    plt.plot(x1,fx1, c = 'black')
    plt.title("LSE")
    plt.grid()
    
    x2 = np.linspace(-6, 6,num=100)
    fx2 = []
    for i in range(len(x2)):
        fxx = 0
        for k in range(2):
            fxx += arr3[k] * (x2[i] ** k)
        fx2.append(fxx)
    
    plt.subplot(3, 1, 2)
    plt.scatter(arr1[0], arr1[1], c ="red", edgecolor = 'black')
    plt.plot(x2, fx2, c = 'black')
    plt.title("Gradient Descent Method")
    plt.grid()
    
    x3 = np.linspace(-6, 6,num=100)
    fx3 = []
    for i in range(len(x3)):
        fxx = 0
        for k in range(n):
            fxx += arr4[k] * (x3[i] ** k)
        fx3.append(fxx)
    
    plt.subplot(3, 1, 3)
    plt.scatter(arr1[0], arr1[1], c ="red", edgecolor = 'black')
    plt.plot(x3, fx3, c = 'black')
    plt.title("Newton's method")
    plt.grid()

    plt.subplots_adjust(hspace=0.6)
    plt.show()    

# Compute the error on LSE
def error(arr1, arr2, n):
    arr1 = arr1.T
    LSE = []
    for i in range(len(arr1)):
        fxx = 0
        for k in range(n):
            fxx += arr2[k] * (arr1[i][0] ** k)
        error = (fxx - arr1[i][1]) ** 2
        LSE.append(error)
    print("Total error: "+str(sum(np.array(LSE))))
    
# LU decomposition to LSE
At = A.T
AtA = dot(At,A)
I = np.eye(n)
lambI = I*lamb
AtA += lambI
L,U = LU(AtA)
inverse_L = Gaussian_Jordan(L)
inverse_U = Gaussian_Jordan(U)
LSE_polynomial = dot(dot(dot(inverse_U, inverse_L), A.T), B)
LSE_polynomial = LSE_polynomial.reshape(-1)
LSE_polynomial = around(LSE_polynomial)

# Gradient Descent
    # initial weights
M = 0
C = 0
    # Training
for _ in range(ITERATION_ROUND):
    Y_hat = M * X_GD + C
    ERROR = - 2 * (Y - Y_hat)
    if M + C >= 0:
        L1_norm = lamb
    else:
        L1_norm = -lamb
    DELTA_M = sum(ERROR * X_GD) / N_sample + L1_norm
    DELTA_C = sum(ERROR) / N_sample + L1_norm

    M -= LR * DELTA_M
    C -= LR * DELTA_C
    
Gradient_polynomial = np.array([C, M])

# Newton's method
AtA = dot(A.T,A)
xn = np.zeros((n,1))
H = 2 * AtA
inv_H = inverse(H)
delta_f = dot(2 * AtA, xn) - 2 * dot(At, B)
N = (dot(inv_H, delta_f)) * (-1)
Newton_polynomial = N.reshape(-1)
Newton_polynomial = around(Newton_polynomial)

# Final result
print("LSE:")
showPolynomial(LSE_polynomial, n)
error(X, LSE_polynomial, n)
print("\n")

print("Gradient Decsent method:")
showPolynomial(Gradient_polynomial, 2)
error(X, Gradient_polynomial, 2)
print("\n")

print("Newton's method:")
showPolynomial(Newton_polynomial, n)
error(X, Newton_polynomial, n)

showResult(X, LSE_polynomial, Gradient_polynomial, Newton_polynomial)