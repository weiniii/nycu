import numpy as np

#generate function
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

#NeuralNetwork
class NeuralNetwork(object):
    import numpy as np
    def __init__(self):
        #parameter
        self.inputsize = 2
        self.outputsize = 1
        self.activation_name = None
        self.learing_rate = 0.01
        self.a = 0.01
        #weight
        self.W1 = np.random.randn(self.inputsize, self.inputsize) #W1 2x2 matrix
        self.W2 = np.random.randn(self.inputsize, self.inputsize) #W2 2x2 matrix
        self.W3 = np.random.randn(self.inputsize, self.outputsize) #W3 2x1 matrix

    def forward(self,x):
        self.xW1 = np.dot(x,self.W1)
        self.z1 = self.activation(self.xW1) #z1
        self.z1W2 = np.dot(self.z1,self.W2)
        self.z2 = self.activation(self.z1W2) #z2
        self.z2W3 = np.dot(self.z2,self.W3)
        output = self.activation(self.z2W3) #y

        return output

    def activation(self,s):
        if self.activation_name == 'Sigmoid':
            return 1.0/(1.0 + np.exp(-s))
        elif self.activation_name == 'ReLU':
            return np.maximum(0.0,s)
        elif self.activation_name == 'LeakyReLU':
            return np.maximum(self.a*s,s)
        elif self.activation_name == 'Tanh':
            return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
        elif self.activation_name == None:
            return s
        else:
            raise NameError
    def derivate_activation(self,s):
        if self.activation_name == 'Sigmoid':
            return np.multiply(s, 1.0 - s)
        elif self.activation_name == 'ReLU':
            s[s<0] = 0
            s[s>=0] = 1
            return s
        elif self.activation_name == 'LeakyReLU':
            s[s<0] = self.a
            s[s>=0] = 1
            return s
        elif self.activation_name == 'Tanh':
            return 1-s**2
        elif self.activation_name == None:
            return 1
        else:
            raise NameError

    def backward(self,x,y,output):
        self.output_loss = (output - y)
        self.output_delta = self.output_loss * self.derivate_activation(output) # W3

        self.z2_loss = self.output_delta.dot(self.W3.T)
        self.z2_delta = self.z2_loss * self.derivate_activation(self.z2) # W2

        self.z1_loss = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z1_loss * self.derivate_activation(self.z1) # W1

        self.W1 -= self.learing_rate*x.T.dot(self.z1_delta)
        self.W2 -= self.learing_rate*self.z1.T.dot(self.z2_delta)
        self.W3 -= self.learing_rate*self.z2.T.dot(self.output_delta)

    def train(self,x,y):
        output = self.forward(x)
        self.backward(x,y,output)

NN = NeuralNetwork()

#generate train data
x,y = generate_linear()

#scientific notation setting
np.set_printoptions(suppress=True)

#training
for i in range(100000):
    NN.train(x,y)
    if(i%5000 ==0):
        print(str(i)+" LOSS: " +str(np.mean(np.square(y - NN.forward(x)))))

#generate test data
testx, testy = generate_linear()

#predict
pred_y = NN.forward(testx)

print("predict:\n"+str(pred_y))

#show result
def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize=18)
    for i in range(testx.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(testx.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
#   Accuracy
s=0
for k in range(100):
    if pred_y[k]>0.5:
        if testy[k]==1:
            s+=1
    else:
        if testy[k]==0:
            s+=1

print(s)
show_result(testx, testy, pred_y)