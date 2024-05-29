from phi.torch.flow import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from env import EnvPDE2
import torch
import numpy as np

torch.manual_seed(0)

rows, cols = [1, 9, 30, 1], [0, -1, -1, 7]

def dataset():
    file_name = 'data'
    po_true = np.load(file_name + '/true_joint.npz')['po'][:, rows, cols]
    po_true = (torch.from_numpy(po_true) - 400)/100
    temp_true = np.load(file_name + '/true_joint.npz')['temp'][:, rows, cols]
    temp_true = (torch.from_numpy(temp_true) - 21)
    print(f'preparing dataset: co2 shape {po_true.shape}, temp shape {temp_true.shape}')
    return po_true, temp_true

def plot_loss(r1, r2):
    fig, ax = plt.subplots(2,1,figsize=(6,5))
    ax[0].plot(r1, label='Learning Loss')
    ax[1].plot(r2, label='Parameter Loss')
    for i in range(2): ax[i].legend()
    plt.savefig('image/simulate_joint_sensor.png', dpi=300)

def train():
    true_po, true_temp = dataset()
    params = {'v': 0.00108, 'k_C': 0.00108,  'k_T':0.002,  'alpha_T': 1.0,  'a': 0.65, 'alpha_C': 200., 'u': 1.2}
    init_params = {'v': 0.002, 'k_C': 2e-3, 'a': 0.6, 'k_T': 0.003, 'alpha_T':1.0, 'alpha_C': 200., 'u': 1.2}
    env = EnvPDE2(init_params, simulator='ct')
    param_group = [
        {'params': [env.p_v, env.p_k_C, env.p_k_T], 'lr': 1e-4},
        {'params': [env.p_a], 'lr': 1e-2} # p_alpha_C, p_alpha_T can be excluded
    ]
    optimizer = optim.Adam(param_group)
    record_loss1, record_loss2 = [], []
    loss_fun = nn.MSELoss()
    for iteration in range(20):
        loss = 0.
        env.init_sys()
        epoch_loss1, epoch_loss2 = 0., 0.
        for t in tqdm(range(60)):
            v, pred_po, pred_temp = env.step(dt=1.0)
            pred_temp = pred_temp.values.native('x,y')[rows, cols]-21.
            pred_po = (pred_po.values.native('x,y')[rows, cols]-400.)/100.
            dloss = loss_fun(pred_temp, true_temp[t]) + loss_fun(pred_po, true_po[t])
            loss += dloss
            epoch_loss1 += dloss.item()
            epoch_loss2 += torch.sum((env.p_v * 10 - 0.00108 * 10) ** 2).item() + torch.sum((env.p_a - 0.65) ** 2).item()\
            + torch.sum((env.p_k_C * 10 - 0.00108 * 10) ** 2).item() +  torch.sum((env.p_k_T * 10 - 0.002 * 10) ** 2).item()\
            + torch.sum((env.p_alpha_T - 1.0) ** 2).item() + torch.sum((env.p_alpha_C/100. - 2.0) ** 2).item()
            if t % 10 == 9:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    env.p_v.clamp_(5e-4, 1e-2)
                    env.p_k_C.clamp_(5e-4, 1e-2)
                loss = 0.
                print(epoch_loss1, epoch_loss2, env.p_v, env.p_k_C, env.p_k_T, env.p_a)
        record_loss1.append(epoch_loss1)
        record_loss2.append(epoch_loss2)
        np.savez('data/record_loss_joint_sensor2.npz', record_l1 = np.array(record_loss1), record_l2 = np.array(record_loss2))
        plot_loss(record_loss1, record_loss2)
        with open(f'data/learn_real2.csv', 'a') as csv_file:
            param = [env.p_v.detach().item(), env.p_a.detach().item(), env.p_k_T.detach().item(), env.p_k_C.detach().item(), env.p_alpha_C.detach().item(), env.p_alpha_T.detach().item()]
            formatted_data = ','.join(str(value) for value in [iteration, epoch_loss1, epoch_loss2]+param) + '\n'
            csv_file.write(formatted_data)


if __name__ == "__main__":
    train()


