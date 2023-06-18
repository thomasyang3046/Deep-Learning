import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, finn_eval_seq, pred

torch.backends.cudnn.benchmark = True #加速cuda計算速度
# In[]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=15, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=20, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=True, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=1, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=2, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args
# In[]
def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()#lstm
    modules['posterior'].zero_grad()#gaussian lstm
    modules['encoder'].zero_grad()#vgg
    modules['decoder'].zero_grad()#vgg
    mse_criterion = nn.MSELoss()
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    x = x.permute(1, 0, 2, 3 ,4).cuda()
    cond = cond.permute(1, 0, 2).cuda()
    h1 = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]#ht-1
    for i in range(1, args.n_past + args.n_future):
        ht = h1[i][0] #ht (encoder right)
        # args.last_frame_skip == True，每次都用前一個frame的skip
        # args.last_frame_skip == False，用最後一張ground truth的frame skip
        if args.last_frame_skip or i < args.n_past:
            ht_1, skip = h1[i-1]
        else:
            ht_1 = h1[i-1][0]#ht-1

        z_t, mu, logvar = modules['posterior'](ht) #Gaussian LSTM get zt
        h_pred = modules['frame_predictor'](torch.cat([cond[i-1], ht_1, z_t], 1)) #LSTM get gt
        x_pred = modules['decoder']([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, args)
        
        #teacher forcing == False, 則用model自己預測frame當作encoder的輸入
        if not use_teacher_forcing :
            h1[i] = modules['encoder'](x_pred)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta #beta = kl_annealing
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)
    
# In[]
#乘KL的係數，佔loss的比率
class kl_annealing():
    def __init__(self, args):
        super().__init__()
        if args.kl_anneal_cyclical:
           self.L = np.ones(args.niter) * 1.0 #存每個epoch的KL annealing factor
           self.i = 0 #record 目前已經計算多少epoch
           period = args.niter / args.kl_anneal_cycle #100//3=33
           step = (1.0 - 0) / (period * 0.5) #1/17.5
           for c in range(args.kl_anneal_cycle): #kl_anneal_cycle=3
               v, i = 0.0, 0.0
               while(v<=1.0 and int(i+c*period)<args.niter):
                   self.L[int(i+c*period)] = v
                   v += step
                   i += 1
        else:
            self.L = np.ones(args.niter) * 1.0
            self.i = 0
            period = (args.niter) / 1
            step = (1.0 - 0) / (period * 0.25) #1/4
            for c in range(1):
                v, i = 0.0, 0.0
                while(v<=1.0 and int(i+c*period)<args.niter):
                    self.L[int(i+c*period)] = v
                    v += step
                    i += 1
     
    def update(self):
        self.i += 1
    
    def get_beta(self):
        return self.L[self.i]
# In[]
def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    print(args.last_frame_skip)
    #n_past = # of frames as input
    #n_futrue = # of frames to predict at training
    #n_eval = # of frames to predict at inference
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30 
    #teacher forcing ratio (0 ~ 1)
    assert 0 <= args.tfr and args.tfr <= 1
    #模型在訓練過程中，從哪個訓練epoch開始降低「老師強制學習比率」
    assert 0 <= args.tfr_start_decay_epoch 
    #如果步幅為 0.1，那麼每個訓練輪數結束時，「老師強制學習比率」將減少 0.1，直到「老師強制學習比率」達到設置的最小值為止。
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load(args.model_dir)
        optimizer = args.optimizer #Adam
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        #rnn_size=256 predictor-posterior-rnn_layers=2-1 n_past=2 n_future=10 g_dim=128 z_dim=64 last_frame_skip=True beta=0.0001
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)
        
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))
    print("build the models")
    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)#g_dim=128
        decoder = vgg_decoder(args.g_dim)#g_dim=128
        encoder.apply(init_weights)#初始化conv, bn, linear
        decoder.apply(init_weights)#初始化conv, bn, linear
    print("transfer to device")
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)#lstm
    posterior.to(device)#gaussian_lstm
    encoder.to(device)#vgg
    decoder.to(device)#vgg
    print("load a dataset")
    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)
    print("optimizers")
    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    print("training loop")
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    args.epoch_size = len(train_loader)
    print(start_epoch)
    
    print(f'epoch size: {args.epoch_size}')
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        print(f'KL beta: {kl_anneal.get_beta()}')
        kl_anneal.update()
        
        #更新teacher forcing ratio，且不低於lower bound的tfr
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            beta = (1.0 - args.tfr_lower_bound) / (args.niter - args.tfr_start_decay_epoch)
            tfr = 1.0 - beta*(epoch - args.tfr_start_decay_epoch) 
            args.tfr = min(max(args.tfr_lower_bound, tfr),1)
        print(f'Teacher ratio: {args.tfr}')
        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        print(f'KL beta: {kl_anneal.get_beta()}')
        kl_anneal.update()
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            slope = (1.0 - args.tfr_lower_bound) / (args.niter - args.tfr_start_decay_epoch)
            tfr = 1.0 - (epoch - args.tfr_start_decay_epoch) * slope
            args.tfr = min(1, max(args.tfr_lower_bound, tfr))
        print(f'Teacher ratio: {args.tfr}')
        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)
                validate_seq = validate_seq.permute(1, 0, 2, 3 ,4).to(device)
                validate_cond = validate_cond.permute(1, 0, 2).to(device)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device) #預測所有frame
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:]) #ground truth 和 prediction的psnr(10 frame)
                psnr_list.append(psnr)
                
            ave_psnr = np.mean(np.concatenate(psnr_list))


            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)
            validate_seq = validate_seq.permute(1, 0, 2, 3 ,4).to(device)
            validate_cond = validate_cond.permute(1, 0, 2).to(device)
            plot_pred(validate_seq, validate_cond, modules, epoch, args)
# In[]
if __name__ == '__main__':
    #using cyclical mode
    main()
