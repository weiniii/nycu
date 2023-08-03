import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import bair_robot_pushing_dataset
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import finn_eval_seq, pred
import matplotlib.pyplot as plt
import torchvision.transforms as T

modules = torch.load('./logs/model.pth')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--data_root', default='./data/', help='root directory for data')
    parser.add_argument('--model_path', default='./logs/model.pth', help='path to model')
    parser.add_argument('--log_dir', default='./git/', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--num_threads', type=int, default=1, help='number of data loading threads')
    parser.add_argument('--nsample', type=int, default=3, help='number of samples')
    parser.add_argument('--N', type=int, default=3, help='number of samples')
    parser.add_argument('--cuda', default=False, action='store_true')
    
    args = parser.parse_args(args=[])
    return args

args = parse_args()
modules = torch.load(args.model_path)
frame_predictor = modules['frame_predictor']
posterior = modules['posterior']

encoder = modules['encoder']
decoder = modules['decoder']

frame_predictor.batch_size = args.batch_size
posterior.batch_size = args.batch_size
args.g_dim = modules['args'].g_dim
args.z_dim = modules['args'].z_dim

device = 'cpu'

frame_predictor.to(device)
posterior.to(device)
encoder.to(device)
decoder.to(device)

# ---------------- set the argsions ----------------
args.last_frame_skip = modules['args'].last_frame_skip

# --------- load a dataset ------------------------------------
test_data = bair_robot_pushing_dataset(args, 'test')

test_loader = DataLoader(test_data,
                         num_workers=args.num_threads,
                         batch_size=args.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)

test_iterator = iter(test_loader)

frame_predictor.eval()
posterior.eval()
encoder.eval()
decoder.eval()

for _ in tqdm(range(1)):
    try:
        test_seq, test_cond = next(test_iterator)
    except StopIteration:
        test_iterator = iter(test_loader)
        test_seq, test_cond = next(test_iterator)

    test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)

    pred_seq = pred(test_seq, test_cond, modules, args, device)
    
for i in range(12):
    pred_seq[i] = pred_seq[i].reshape(3,64,64)
    
transform = T.ToPILImage()

plt.figure(figsize=(40, 400))
for i in range(1,13):
    plt.subplot(1,12,i)
    plt.imshow(transform(pred_seq[i-1]))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0)
plt.show()
plt.savefig('prediction.png')
test = []
for i in range(12):
    t = test_seq[i].reshape(3,64,64)
    test.append(t)

plt.figure(figsize=(40, 400))
for i in range(1,13):
    plt.subplot(1,12,i)
    plt.imshow(transform(test[i-1]))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0)
plt.show()
plt.savefig('gound_truth.png')
