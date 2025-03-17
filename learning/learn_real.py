from env import EnvPDE
import torch
import torch.nn.functional as F
from torch import nn, autograd
import torch.optim as optim
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import time 
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--c', type=int, default=1)
args = parser.parse_args()
c = args.c

torch.manual_seed(0)

def dataset():
    data = io.loadmat('data/1HourCycle0929.mat') 
    s = 1793-60*5 + 3600
    x = list(range(s, s+195*60)) # train on this sequence
    d_return = (data['return_ori'][x][::60] - 400) / 100.
    d_supply = (data['supply_ori'][x][::60] - 400) / 100.
    d_black = (data['black_ori'][x][::60] - 400) / 100.
    d_true = torch.from_numpy(np.concatenate((d_return, d_supply, d_black), axis=1))
    return d_true


params =  {'v': 0.002, 'k_C':0.002, 'a': 0.6, 'alpha_C': 800.0, 'u': 1.2}
env = EnvPDE(params, plot=False)
param_group = [
    {'params': [env.p_v, env.p_k_C], 'lr': 1e-4},
    {'params': [env.p_a], 'lr': 1e-3}
]
optimizer = optim.Adam(param_group)


def train():
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    d_true = dataset()
    return_rows, return_cols =  [10, 9], [-1,-1]
    supply_rows, supply_cols =[30],  [-1]
    black_rows, black_cols = [1, 2], [7, 8]
    record_loss = []
    time_s = time.time()
    record_loss_t = []
    for iteration in range(20):
        scheduler.step()
        loss, epoch_loss = 0., 0.
        env.init_sys()
        env.Pocoeff = [0] * 15 + ([1.0] * 30 + [0] * 30) * 3
        env.u = [1.0] * len(env.Pocoeff)
        for t in tqdm(range(140)):
            velocity, po, pressure = env.step()
            env.save('train_t90_update_all')
            pred_return = (po.values.native('x,y')[return_rows, return_cols] - 400 ) / 100.
            pred_supply = (po.values.native('x,y')[supply_rows, supply_cols] - 400 ) / 100.
            pred_black = (po.values.native('x,y')[black_rows, black_cols] - 400 ) / 100.
            if t > 40:  # warm
                loss_t =  torch.sum((pred_return.mean() - d_true[t][0])**2) +  torch.sum((pred_black.mean() - d_true[t][2])**2) +  torch.sum((pred_supply.mean() - d_true[t][1])**2)
                loss += loss_t
                epoch_loss += loss_t.item()
                record_loss_t.append(loss_t.item())
            if t > 40 and t % 140 == 139:
                time_e = time.time()
                param = [env.p_v.detach().item(), env.p_a.detach().item(), env.p_alpha_C.detach().item(), env.p_k_C.detach().item(), env.p_u.detach().item()]
            if t > 40 and t % 140 == 139:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                   env.p_v.clamp_(5e-4, 1e-2)
                   env.p_k_C.clamp_(5e-4, 1e-2)
                loss = 0.
        record_loss.append(epoch_loss)
        np.savez(f'record_loss_t_{c}.npz', loss = np.array(record_loss_t))
        with open(f'data/learn_co2.csv', 'a') as csv_file:
            formatted_data = ','.join(str(value) for value in [iteration, epoch_loss]+param) + '\n'
            csv_file.write(formatted_data)

def evaluate():
    params = {'v': 0.00257834, 'k_C':0.0010025, 'a': 0.6147829, 'alpha_C': 800.0, 'u': 1.2} # trained value
    env = EnvPDE(params, plot=False)
    env.init_sys()
    env.Pocoeff = [0] * 20 + ([1.0] * 30 + [0] * 30) * 5  + [0] * int(2.5*60+10)
    env.u = [1.0] * len(env.Pocoeff)
    for t in tqdm(range(len(env.u))):
        velocity, po, pressure = env.step()
        env.save('val')

if __name__ == "__main__":
    train()
    # evaluate()
