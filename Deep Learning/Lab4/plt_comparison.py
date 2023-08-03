import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

mopsnr = torch.load('./cyclic/avgpsnr.pt')
moklweig = torch.load('./cyclic/kl_weight.pt')
moloss = torch.load('./cyclic/loss.pt')
motfr = torch.load('./cyclic/tfr.pt')

epoch = []
for i in range(1,101):
    epoch.append(i)
epoch = np.array(epoch)    
psnr_epoch = []
for i in range(1,101):
    if i%5==0 or i==1:
        psnr_epoch.append(i)
psnr_epoch = np.array(psnr_epoch)   

fig, ax1 = plt.subplots()
plt.title('Training loss/ratio curve')
plt.xlabel('epochs')
ax2 = ax1.twinx()
ax1.set_ylabel('loss/psnr')
ax1.plot(epoch, moloss, 'b', label='total loss')
ax1.plot(psnr_epoch, mopsnr, 'g.', label='psnr')
ax1.legend()
ax1.set_ylim([0.0, 30.0])
ax2.set_ylabel('ratio')
ax2.plot(epoch, motfr, 'm--', label='Teacher ratio')
ax2.plot(epoch, moklweig, '--', color='orange', label='KL weight')
fig.tight_layout()
ax2.legend()
plt.savefig('cyclic.png')

