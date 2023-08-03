import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

# data
x, y, testx, testy = read_bci_data()
x = torch.from_numpy(x)
y = torch.from_numpy(y)
testx = torch.from_numpy(testx)
testy = torch.from_numpy(testy)
x=x.float() 
y=y.float() 
testx=testx.float() 
testy=testy.float() 

# model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=(1,39), stride=(1,1), padding =(0,19), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            
            # Layer 2
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,5), stride=(1,5)),
            nn.Dropout(p=0.5),

            # Layer 3
            nn.Conv2d(32, 32, kernel_size=(1,39), stride=(1,1), padding=(0,19), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
            nn.Dropout(p=0.5),
        )    
        self.fc = nn.Sequential(
            nn.Linear(in_features=1184, out_features=2, bias=True)
        )
    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
    
# setting
device = "cuda" if torch.cuda.is_available() else "cpu"
Epoahs = 300
batch_size = 128
b = lambda batch_size:1 if 1080%batch_size else 0
b = b(batch_size)
model = EEGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

# training and testing
trainp = []
testp = []
for epoch in tqdm(range(Epoahs)):
    model.train()
    p = []
    for i in range(int(len(x)/batch_size)+b):
        s = i*batch_size
        if i != int(len(x)/batch_size):
            e = (i+1)*batch_size
        else:
            e = i*batch_size + len(x)%batch_size
        inputs = x[s:e]
        labels = y[s:e]
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        pp = np.argmax(pred.cpu().detach().numpy(),axis=1)
        p = np.concatenate([p,pp])
    a =  accuracy_score(y.cpu().numpy(),p)       
    trainp.append(a)
        
    model.eval()
    p = [] 
    for i in range(int(len(testx)/batch_size)+b):
        s = i*batch_size
        if i != int(len(testx)/batch_size):
            e = (i+1)*batch_size
        else:
            e = i*batch_size + len(x)%batch_size
        inputss = testx[s:e]
        labelss = testy[s:e]
        labelss = labelss.type(torch.LongTensor)
        inputss, labelss = inputss.to(device), labelss.to(device)

        with torch.no_grad():
            predd = model(inputss)
            loss = criterion(predd,labelss)
            pp = np.argmax(predd.cpu().detach().numpy(),axis=1)
            p = np.concatenate([p,pp])
    a = accuracy_score(testy.cpu().numpy(),p)    
    testp.append(a)
print(a)