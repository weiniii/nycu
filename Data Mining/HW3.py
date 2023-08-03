import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy.stats import entropy
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans

METHOD = 'KNN'

def save(result):
      final = np.empty((len(result), 2)).astype(object)
      for i in range(len(final)):
            final[i, 0] = str(i)
            final[i, 1] = str((result[i]))

      result_df = pd.DataFrame(final, columns = ['id','outliers'])
      result_df.to_csv('STUDENT_ID.csv', index=None)

def Calculate_mse_loss(test, center):
    final = []
    for i in range(len(test)):
        final.append(torch.mean(torch.square((center - test[i])), dim=1)[torch.where(torch.mean(torch.square((center - test[i])), dim=1) == torch.min(torch.mean(torch.square((center - test[i])), dim=1)), 1, 0).nonzero()[0].item()].item())
    
    return final

def Show_AUC_score(truth_table, final):
    fpr, tpr, thresholds = roc_curve(truth_table, final)
    plt.plot(fpr, tpr)
    print(roc_auc_score(truth_table, final))
    return roc_auc_score(truth_table, final)

train_df = pd.read_csv(r'training.csv')
test_df = pd.read_csv(r'test_X.csv')
sample_df = pd.read_csv(r'sample.csv')

x = train_df.values[: ,1:].astype(float)
x = torch.Tensor(x)
category = train_df.values[:, 0]
embedding = {list(set(category))[i]:i for i in range(len(set(category)))}
y = [embedding[category[i]] for i in range(len(category))]
y = torch.LongTensor(y)

sample_label = sample_df.values[:, 0]
sample_data = sample_df.values[:, 1:].astype(float)

test = torch.Tensor(test_df.values.astype(float))

sample_data = torch.Tensor(sample_data)
sample_data_truth = [0 if sample_label[i] in category else 1 for i in range(len(sample_data))]

train_data , test_data , train_label , test_label = train_test_split(x, y, test_size=0.01, shuffle=True)


if METHOD == 'KNN':
    Kmeans = KMeans(n_clusters=800, n_init='auto')

    Kmeans.fit(x)

    center = Kmeans.cluster_centers_
    center = torch.Tensor(center)

    def Calculate_mse_loss(test, center):
        final = []
        for i in range(len(test)):
            final.append(torch.mean(torch.square((center - test[i])), dim=1)[torch.where(torch.mean(torch.square((center - test[i])), dim=1) == torch.min(torch.mean(torch.square((center - test[i])), dim=1)), 1, 0).nonzero()[0].item()].item())

        return final


    final = Calculate_mse_loss(sample_data, center)
    Show_AUC_score(sample_data_truth, final)
    returns = Calculate_mse_loss(test, center)

elif METHOD == "Autoencoder":

    epochs = 20
    batch_size = 8
    eps = 1e-5
    learning_rate = 1e-3
    show = False

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128)
            )
            self.decoder = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )
        def forward(self, x):
            z = self.encoder(x)
            output = self.decoder(z)
            return output

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=eps)
    train_loader = DataLoader(TensorDataset(x, y), batch_size, shuffle=True)

    sample_loss_list = []
    test_loss_list = []
    train_loss_list = []
    for iteration in tqdm(range(epochs)):
        model.train()
        for sub_x, sub_y in train_loader:
            
            pred = model(sub_x)
            loss = F.mse_loss(pred, sub_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        model.eval()
        train_pred = model(x)
        train_loss = F.mse_loss(train_pred, x)
        train_loss_list.append(train_loss.detach().numpy())

        test_pred = model(test)
        test_loss = F.mse_loss(test_pred, test)
        test_loss_list.append(test_loss.detach().numpy())
        
        sample_pred = model(sample_data)
        sample_loss = F.mse_loss(sample_pred, sample_data)
        sample_loss_list.append(sample_loss.detach().numpy())

    if show:
        x1 = [i for i in range(len(sample_loss_list))]
        plt.plot(x1, sample_loss_list, label='sample')
        plt.plot(x1, test_loss_list, label='test')
        plt.plot(x1, train_loss_list, label='train')
        plt.legend()

    result = model(sample_data)
    final_loss = torch.mean(torch.square(result - sample_data), dim=1).detach().numpy()
    returns = Show_AUC_score(sample_data_truth, final_loss)





satisfy = False


if satisfy:
    save(result)