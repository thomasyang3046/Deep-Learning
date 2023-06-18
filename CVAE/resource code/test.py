import torch
#import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
#sys.path.append('C:/Users/jimmy/OneDrive/桌面/交大1112/深度學習實習/HW5/sample_code')
import utils
import itertools
from tqdm import tqdm
import numpy as np

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import finn_eval_seq, pred,plot_pred
#import os
#os.chdir('C:/Users/jimmy/OneDrive/桌面/交大1112/深度學習實習/HW5/sample_code')

# In[]
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--data_root', default='processed_data', help='root directory for data')
parser.add_argument('--model_path', default='logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/JJLin.pth', help='path to model')
parser.add_argument('--log_dir', default='./logs/fp', help='directory to save generations to')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=1, help='number of data loading threads')
args = parser.parse_args()
os.makedirs('%s' % './logs/fp', exist_ok=True)
args.n_eval = args.n_past+args.n_future

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
dtype = torch.cuda.FloatTensor
modules = torch.load('C:/Users/jimmy/OneDrive/桌面/交大1112/深度學習實習/HW5/sample_code/logs/fp/rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/JJLIN.pth')
frame_predictor = modules['frame_predictor']
posterior = modules['posterior']


encoder = modules['encoder']
decoder = modules['decoder']


frame_predictor.batch_size = args.batch_size
posterior.batch_size = args.batch_size
args.g_dim = modules['args'].g_dim
args.z_dim = modules['args'].z_dim
frame_predictor.cuda()
posterior.cuda()
encoder.cuda()
decoder.cuda()
args.last_frame_skip = modules['args'].last_frame_skip
    # --------- load a dataset ------------------------------------
#sys.path.append('C:/Users/jimmy/OneDrive/桌面/交大1112/深度學習實習/HW5/sample_code')
test_data = bair_robot_pushing_dataset(args, 'test')

test_loader = DataLoader(test_data,
                         num_workers=args.num_threads,
                         batch_size=args.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)
# In[]
def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px
# In[]
print("start test")
device = 'cuda'
frame_predictor.eval()
posterior.eval()
encoder.eval()
decoder.eval()
psnr_list = []

for i, (test_seq, test_cond) in enumerate(tqdm(test_loader)):

    test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)
    pred_seq = pred(test_seq, test_cond, modules, args, device)
    _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
    psnr_list.append(psnr)
ave_psnr = np.mean(np.concatenate(psnr_list))
print(f'====================== test psnr = {ave_psnr:.5f} ========================')
# In[]
test_iterator = iter(test_loader)
for i in range(256):
    test_seq, test_cond = next(test_iterator)
    test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)
    plot_pred(test_seq, test_cond, modules, i, args)
    
  
# In[]
test_iterator = iter(test_loader)
test_seq, test_cond = next(test_iterator)
test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
test_cond = test_cond.permute(1, 0, 2).to(device)
idx=0
frame_predictor.hidden = frame_predictor.init_hidden()
posterior.hidden = posterior.init_hidden()
posterior_gen = []
posterior_gen.append(test_seq[0])
x_in = test_seq[0]
for i in range(1, args.n_eval):
    ht_1 = encoder(x_in)
    ht = encoder(test_seq[i])[0].detach()
    if args.last_frame_skip or i < args.n_past:	
        ht_1, skip = ht_1
    else:
        ht_1, _ = ht_1
    ht_1 = ht_1.detach()
    _, z_t, _= posterior(ht) #Gaussian LSTM get zt
    if i < args.n_past:
        frame_predictor(torch.cat([test_cond[i-1], ht_1, z_t], 1)) 
        posterior_gen.append(test_seq[i])
        x_in = test_seq[i]
    else:
        h_pred = frame_predictor(torch.cat([test_cond[i-1], ht_1, z_t], 1)).detach()
        x_in = decoder([h_pred, skip]).detach()
        posterior_gen.append(x_in)# 得到預測輸出
  
nsample = 3
psnr = np.zeros((args.batch_size, nsample, args.n_future))
progress = tqdm(total=nsample)
all_gen = []
for s in range(nsample):
    progress.update(1)
    gen_seq = []
    gt_seq = []
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    x_in = test_seq[0]
    all_gen.append([])
    all_gen[s].append(x_in)
    for i in range(1, args.n_eval):
        h = encoder(x_in)
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        if i < args.n_past:
            ht = encoder(test_seq[i])[0].detach()
            _, z_t, _ = posterior(ht)
        else:
            z_t = torch.randn(args.batch_size, args.z_dim).cuda()
        if i < args.n_past:
            frame_predictor(torch.cat([test_cond[i-1], h, z_t], 1))
            x_in = test_seq[i]
            all_gen[s].append(x_in)
        else:
            h = frame_predictor(torch.cat([test_cond[i-1], h, z_t], 1)).detach()
            x_in = decoder([h, skip]).detach()
            gen_seq.append(x_in)
            gt_seq.append(test_seq[i])
            all_gen[s].append(x_in)
    _, _, psnr[:, s, :] = finn_eval_seq(gt_seq, gen_seq)

for i in range(args.batch_size):
    gifs = [ [] for t in range(args.n_eval) ]
    text = [ [] for t in range(args.n_eval) ]
    mean_psnr = np.mean(psnr[i], 1)
    ordered = np.argsort(mean_psnr)
    rand_sidx = [np.random.randint(nsample) for s in range(3)]
    for t in range(args.n_eval):
        # gt 
        gifs[t].append(add_border(test_seq[t][i], 'green'))
        text[t].append('Ground\ntruth')
        #posterior 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(posterior_gen[t][i], color))
        text[t].append('Approx.\nposterior') #靠zt生成
        # best 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        sidx = ordered[-1]
        gifs[t].append(add_border(all_gen[sidx][t][i], color))# 隨機zt生成，挑PSNR最好的
        text[t].append('Best PSNR')
        # random 3 ，隨機zt去生成
        for s in range(len(rand_sidx)):
            gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
            text[t].append('Random\nsample %d' % (s+1))

    fname = '%s/%s_%d.gif' % (args.log_dir, 'test', idx+i) 
    utils.save_gif_with_text(fname, gifs, text)
