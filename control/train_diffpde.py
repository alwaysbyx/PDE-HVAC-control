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
from torch.utils.tensorboard import SummaryWriter
from phi.torch.flow import math
from phi.torch.flow import functional_gradient
import argparse



math.seed(0)


occu = np.zeros(360)
occu[10:30] = 6
occu[40:60] = 6
occu[100:140] = 8
occu[200:320] = 3
occu[290:320] = 5
T = 360
max_epoch = 30
ratio = 0.1

def control_loss(u, u_air_t = 21.0 * np.ones(T)):
    params = {'v': 0.00108, 'k_C': 0.00108,  'k_T':0.002,  'alpha_T': 0.3,  'a': 0.65, 'alpha_C': 50.}
    params['u'] = u
    params['u_air_t'] = u_air_t
    params['sT'] = 60
    params['M'] = 1.4
    params['v_df'] = "v0"
    env = EnvPDE(params, simulator='c')
    env.Pocoeff = occu
    loss = 0.0
    for t in tqdm(range(T)):
        v, po = env.step(dt=1.0)
        state_loss = -0.1 * math.log(3-po.values.max/400.)
        vio_loss = 0.
        loss += state_loss + vio_loss
    loss += math.l1_loss(u[0:]) + 0.5*math.sum(math.abs(u[1:]-u[:-1]))
    return loss, env

def control_loss2(u, u_air_t = 21.0 * np.ones(T)):
    params = {'v': 0.00108, 'k_C': 0.00108,  'k_T':0.002,  'alpha_T': 0.3,  'a': 0.65, 'alpha_C': 50.}
    params['u'] = u
    params['u_air_t'] = u_air_t
    params['sT'] = 60
    params['M'] = 1.4
    params['v_df'] = "v0"
    env = EnvPDE(params, simulator='t')
    env.Pocoeff = occu
    env.temp_out = np.load('data/sf_out_temp.npz')['out_temp']
    loss = 0
    T = 360
    for t in tqdm(range(T)):
      v, temp = env.step(dt=1.0)
      vio_loss = -0.2 * math.log(22.0-temp.values.mean)
      vio_loss += -0.2 * math.log(temp.values.mean-21.0)
      loss += vio_loss
    loss += math.l1_loss(u_air_t[0:]-14.44)
    return loss, env

def train_airflow_rate(u, max_epoch):
    u = u.copy()
    u_air_t = 21.0 * np.ones(T)
    lr = 0.1
    for epoch in range(max_epoch):
        grad_fun = functional_gradient(control_loss, 'u', get_output=True)
        if epoch == 10:
            lr = 0.01
        elif epoch == 3:
            lr = 0.03
        (loss, env), dx = grad_fun(u=u, u_air_t=u_air_t)
        print(loss, dx)
        np.savez(f'data/result_u1_epoch{epoch}.npz',  po=np.array(env.pos), temp = np.array(env.temps), occu=np.array(env.Pocoeff), u=env.p_u.detach().numpy(), u_air_t = env.u_airt, epoch_loss = loss.numpy())
        u[:] -= lr * dx[:].detach().numpy()
        u = np.clip(u, 0.1, 1.)
    
    
def train_airflow_temp(max_epoch=30):
    u = np.load(f'data/result_u1_epoch{max_epoch-1}.npz')["u"]
    u_air_t = 21.0 * np.ones(360)
    lr = 0.1
    for epoch in range(max_epoch):
        if epoch == 5: lr = 1e-1
        elif epoch == 10: lr = 0.05
        elif epoch == 15: lr = 0.01
        grad_fun = functional_gradient(control_loss2, 'u_air_t', get_output=True)
        (loss, env), dx = grad_fun(u, u_air_t)
        print(loss, dx)
        np.savez(f'data/result_u12_epoch{epoch}.npz',  po=np.array(env.pos), temp = np.array(env.temps), occu=np.array(env.Pocoeff), u=env.p_u, u_air_t = env.u_airt.detach().numpy(), epoch_loss = loss.numpy())
        u_air_t -= lr * dx.detach().numpy()
        u_air_t = np.clip(u_air_t, 14.44, 22.0)


if __name__ == "__main__":
    u = np.array([0.5] * 60)
    train_airflow_rate(u, max_epoch=30)
    train_airflow_temp(max_epoch=30)



