import numpy as np
import matplotlib.pyplot as plt

s = input("Input your question 1.a, 1.b, 2, or 3: ")
def gaussian(m, s):
    sample = np.sum(np.random.uniform(0, 1, 12)) - 6
    output = m + sample * s ** (1 / 2)
    
    return output

def Polynomial_generator(n, a, w):
    
    e = gaussian(0, a)
    x = np.random.uniform(-1, 1)
    phi_x = np.array([x ** i for i in range(n)])
    y = phi_x @ w + e
        
    return x, y

### 1. Random Data Generator

### a. Univariate gaussian data generator
if s == '1.a':
    m, s = map(float, input("Input m s: ").split(' '))
    sample = gaussian(m, s)
    print(sample)

### b. Polynomial basis linear model data generator
elif s == '1.b':
    

    while True:
        inputs = list(map(float, input("Input n a w: ").split(' ')))
        n = int(inputs[0])
        a = inputs[1]
        w = np.array(inputs[2:])
        if n == len(w):
            print('Success')
            break
        else:
            print("Input error")
                
    x, y = Polynomial_generator(n, a, w)
    print(f"x: {x}  y: {y}")


### 2. Sequential Estimator
elif s == '2':
    m, s = map(float, input("Input m s: ").split(' '))

    times = 1
    mu_ = 0
    var_ = 0
    old_var_ = 0
    print(f"Data points distribution: N({m}, {s})\n")

    while True:
        add_data = gaussian(m, s)
        mu_new = (mu_ * (times - 1) + add_data) / times
        var_ = (var_ * (times - 1) + mu_ ** 2 * (times - 1) + add_data ** 2 ) / times - mu_new ** 2
        
        
        print(f"Add data point: {add_data:15.10f}")
        print(f"Mean: {mu_new:15.10f} , Var: {var_:15.10f}\n")
        
        if abs(old_var_ - var_) < 1e-7 and times > 1000:
            print(f"times: {times}")
            break
        
        old_var_ = var_
        mu_ = mu_new
        times += 1

### 3. Baysian Linear regression
elif s == '3':
    def Print(arr):
        for rows in arr:
            print(''.join([f'{element:>10.5f}' for element in rows]))
            
    def Plot_result(id, title, w, var, x, y):
        
        plt.subplot(id)
        plt.xlim(-2.0, 2.0)
        plt.ylim(-23.0, 23.0)
        plt.title(title)
        
        if x:
            plt.scatter(x, y, s=10)
            
        function = np.poly1d(np.flip(w))
        x_curve = np.linspace(-2.0, 2.0, 100)
        y_curve = function(x_curve)
        
        
        if title == 'After 10 incomes' or title == 'After 50 incomes':
            var_ = np.zeros(100)
            for j, x in enumerate(x_curve):
                X = np.array([x ** i for i in range(len(w))]).reshape(1, len(w))
                
                var_[j] = X @ var @ X.T
            
            var = var_
        
        plt.plot(x_curve, y_curve, 'k')
        plt.plot(x_curve, y_curve + var + 1, 'r')
        plt.plot(x_curve, y_curve - var - 1, 'r')
        
    while True:
        inputs = list(map(float, input("Input b n a w: ").split(' ')))
        b = int(inputs[0])
        n = int(inputs[1])
        ground_a = inputs[2]
        ground_w = np.array(inputs[3:])
        if n == len(ground_w):
            print('Success')
            break
        else:
            print("Input error")

    x = []
    y = []

    times = 0
    m = np.zeros(n)
    var_ = 0

    var = np.eye(n, dtype=float) * 1/b
    mean = np.zeros((n, 1), dtype=float)
    
    var_predictive = 0
    old_var_predictive = 0

    while True:
        add_x, add_y = Polynomial_generator(n, ground_a, ground_w)
        print(f"Add data point ({add_x}, {add_y}): ")
        
        times += 1
        
        x.append(add_x)
        y.append(add_y)
        
        a = ground_a
        X = np.array([add_x ** i for i in range(n)]).reshape(1, n)
        
        if times == 1:
            SIGMA = X.T @ X * a + np.eye(n) * b
            inv_SIGMA = np.linalg.inv(SIGMA)
            m = inv_SIGMA @ X.T * add_y * a
        else:
            NEW_SIGMA = X.T @ X * a + SIGMA
            inv_SIGMA = np.linalg.inv(NEW_SIGMA)
            m = inv_SIGMA @ (X.T * add_y * a + SIGMA @ m)
            SIGMA = NEW_SIGMA
            
        
        m_predictive = X @ m
        var_predictive = X @ inv_SIGMA @ X.T + 1 / a

        print("\n Posterior mean:")
        Print(m)
        print("\n Posterior variance:")
        Print(inv_SIGMA)
        print(f"\n Predictive distribution ~ N({m_predictive.item():6.4f}, {var_predictive.item():6.4f})")
        
        print('-------------------------------------------------------------')
        
        if abs(old_var_predictive - var_predictive) < 1e-9 and times > 1000:
            m_prediction = m.reshape(-1)
            var_prediction = var_predictive.item()
            break
        
        old_var_predictive = var_predictive

        if times == 10:
            m_10 = m.reshape(-1)
            # var_10 = var_predictive.item()
            var_10 = inv_SIGMA
        elif times == 50:
            m_50 = m.reshape(-1)
            # var_50 = var_predictive.item()
            var_50 = inv_SIGMA
            
        else:
            pass


    ids = [221, 222, 223, 224]
    title = ['Ground truth', 'Predict Result', 'After 10 incomes', 'After 50 incomes']

    w_list = [ground_w, m_prediction, m_10, m_50]
    var_list = [ground_a, var_prediction, var_10, var_50]
    x_list = [None, x, x[:10], x[:50]]
    y_list = [None, y, y[:10], y[:50]]

    for i in range(4):
        Plot_result(ids[i], title[i], w_list[i], var_list[i], x_list[i], y_list[i])

    plt.tight_layout()
    plt.show()

else:
    print("Input error")