from phi.flow import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from phiml.math import reshaped_numpy
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id='BuildingEnv-v0',
    entry_point='RLenv:RLEnv',
)


Test_Pocoeff = [[0] * 10 + [7.0] * 20 + [0] * 10 + [6.0] * 20 + [0] * 20, \
                [0] * 10 + [6.0] * 20 + [0] * 10 + [6.0] * 20 + [0] * 15, \
                [0] * 10 + [5.0] * 20 + [0] * 10 + [7.0] * 25 + [0] * 35, \
                [0] * 10 + [8.0] * 20 + [0] * 10 + [8.0] * 25 + [0] * 35, \
                [0] * 10 + [5.0] * 20 + [0] * 10 + [8.0] * 25 + [0] * 35, \
                [0] * 10 + [4.0] * 20 + [0] * 10 + [5.0] * 25 + [0] * 35]

class RLEnv(gym.Env):
  def __init__(self, params):
    super(RLEnv, self).__init__()
    self.action_space = spaces.Box(low=np.array([0.1, 18.44]), high=np.array([1.0, 21.0]), shape=(2,), dtype=np.float32)
    self.observation_space = spaces.Box(low=0., high=10.0, shape=(3,), dtype=np.float32) 
    self.p_v = params['v']
    self.p_a = params['a']
    self.p_alpha_C = params['alpha_C']
    self.p_k_C = params['k_C']
    self.p_alpha_T = params['alpha_T']
    self.p_k_T = params['k_T']
    self.save = params['save']
    if self.save:
      self.pos = []
      self.temps = []
    
  def step(self, action):
    po = self.step_co2(action, dt=1.0)
    temp = self.step_temperature(action, dt=1.0)
    self.po, self.temp = po, temp
    v, p = self.v, self.p
    v = advect.semi_lagrangian(v, v, 1.0)
    v = diffuse.explicit(v, self.p_v, 1.0, substeps=4)
    v, p = fluid.make_incompressible(v, (self.ob1, self.ob2, self.ob3), Solve(x0=p))
    airflow = v.with_values(lambda x, y: tensor(vec(x=0, y= - action[0] * 1.4 *(x<=3.3)*(x>=2.7))))
    v = field.where(self.airflow_pos, airflow, v)
    self.v = v
    self.p = p
    self.t += 1
    if self.t != 60:
      observation = np.array([(self.po.values.max-400.)/400.0, self.temp.values.mean / 21.0, self.Pocoeff[self.t]/10.])# np.stack((self.po.values.numpy('x,y'), self.temp.values.numpy('x,y')), axis=2)
    else:
      observation = np.array([(self.po.values.max-400.)/400.0, self.temp.values.mean / 21.0, self.Pocoeff[self.t-1]/10.])
    done = self.t >= 60
    if po.values.max > 1200 or temp.values.mean < 21.0: done = True
    info = {}
    reward = -self.loss_fun(action[0], action[1], po, temp)
    if done: reward += self.t
    return observation, reward, done, False, info 

  def loss_fun(self, u1, u2, po, temp):
    state_loss = 0.1 * np.mean((po.values.max/400-1.)**2)
    vio_loss = np.sum(np.maximum(po.values.max-1200, 0. * po.values.max))
    vio_loss += 100.0*np.sum(np.maximum(temp.values.mean-24.0, 0. * temp.values.max))
    vio_loss += 100.0*np.sum(np.maximum(21.0 - temp.values.mean, 0. * temp.values.min))
    control_loss = np.abs(u1) + 0.1 * np.abs(u2-14.44) 
    return state_loss + vio_loss + control_loss

  def step_co2(self, action, dt=1.):
    v, po, p = self.v, self.po, self.p
    po_fresh = po.with_values(lambda x, y:  400.0*self.p_a+ po.values.mean*(1-self.p_a)*(x>=0))
    po = field.where(self.hvac_pos, po_fresh, po)
    po_in = self.p_alpha_C * self.Pocoeff[self.t] * resample(Box(x=(2.5, 3.1), y = (0.7,0.9)), to=po, soft=True)
    po = advect.mac_cormack(po, v, dt) + po_in
    po = diffuse.explicit(po, self.p_k_C, dt, substeps=4)
    po = math.stop_gradient(po)
    if self.save: self.pos.append(po.values.native('x,y'))
    return po

  def step_temperature(self, action, dt=1.):
    v, temp, p = self.v, self.temp, self.p
    temp_in = self.p_alpha_T * self.Pocoeff[self.t] * resample(Box(x=(2.5, 3.1), y = (0.7,0.9)), to=temp, soft=True) 
    temp = advect.mac_cormack(temp, v, dt) + temp_in
    temp = diffuse.explicit(temp, self.p_k_T, dt, substeps=4)
    temp_fresh = temp.with_values(lambda x, y:  action[1] * (x>=0))
    temp = field.where(self.hvac_pos, temp_fresh, temp)
    temp = math.stop_gradient(temp)
    if self.save: self.temps.append(temp.values.native('x,y'))
    return temp

  def reset(self, seed=None, options=None):
    c0 = 400.
    t0 = 22.0
    self.v =  StaggeredGrid(0, extrapolation.combine_sides(x=(0., 0.), y=(0., ZERO_GRADIENT)), x=42, y=27, bounds=Box(x=4.2, y=2.7))
    self.po = CenteredGrid(c0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
    self.hvac_pos = self.po.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
    self.temp = CenteredGrid(t0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
    self.airflow_pos = self.v.with_values(lambda x, y: vec(x=0, y=y>=2.65))
    self.ob1 = Obstacle(Box(x=(1.5, 2.7), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
    self.ob2 = Obstacle(Box(x=(0, 0.9), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
    self.ob3 = Obstacle(Box(x=(3.3, 4.2), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
    self.p = None
    schedule = 1 # int(np.random.uniform(0, 6))
    self.Pocoeff = Test_Pocoeff[schedule]
    self.t = 0
    observation = np.array([(self.po.values.max-400.)/400.0, self.temp.values.mean / 21.0, 0.0])# np.stack((self.po.values.numpy('x,y'), self.temp.values.numpy('x,y')), axis=2)
    return observation, {}

  def render(self):
    pass 

