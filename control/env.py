from phi.torch.flow import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from phiml.math import reshaped_numpy
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
import torch

cmap=sns.color_palette("vlag", as_cmap=True)
cmap=sns.color_palette("Blues", as_cmap=True)

class EnvPDE:
    def __init__(self, params, simulator='c', plot=False):
        self.simulate_t = 't' in simulator
        self.simulate_c = 'c' in simulator
        self.p_v = torch.tensor(params['v'])
        self.p_a = torch.tensor(params['a'])
        self.p_u = params['u']
        self.u_airt = params['u_air_t']
        self.v_df = params.get('v_df')
        self.po_df = params.get('po_df')
        if self.simulate_c:
            self.p_alpha_C = torch.tensor(params['alpha_C'])
            self.p_k_C = torch.tensor(params['k_C'])
        if self.simulate_t:
            self.p_alpha_T = torch.tensor(params['alpha_T'])
            self.p_k_T = torch.tensor(params['k_T'])
        self.init_plot(plot)
        self.init_sys()
        self.pos = []
        self.temps = []
        self.sT = params['sT']
        self.M = params['M']

    def init_plot(self, plot):
        self.plot_flag = plot
        if plot:
            self.rows, self.cols = [28, 10, 2], [-1, -1, 15]
            self.fig, self.ax = plt.subplots(figsize=(8,6))
            plt.rc('font', size=14)

    def update_plot(self):
        data = self.v.at_centers()
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        x, y = reshaped_numpy(data.points.vector[0,1], [vector, data.shape.without('vector')])
        u, v = reshaped_numpy(data.values.vector[0,1], [vector, extra_channels, data.shape.without('vector')])
        for ch in range(u.shape[0]):
            self.ax.quiver(x, y,  u[ch], v[ch], color="black")
        po = self.temp.values.native('y,x').detach().numpy() # [::-1, :]
        x = np.linspace(0, 4.2, 42)
        y = np.linspace(0, 2.7, 27)
        X, Y = np.meshgrid(x,y)
        self.ax.imshow(po[::-1], extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, interpolation_stage='rgba')
        self.ax.set_xlabel('x', fontsize=14)
        self.ax.set_ylabel('y', fontsize=14)
        #im = self.ax.imshow(po, origin='lower', vmin=400., vmax=2000.)
        #self.cb = plt.colorbar(im, ax=self.ax)
        plt.savefig(f'image/{self.t}.png', dpi=300)
        plt.pause(1e-3)
        #self.cb.remove()
        self.ax.clear() 

    def save(self):
        if self.t % 10 == 9:
            np.savez('data/simulate_train.npz', c=np.array(self.pos))

    def step_co2(self, dt=1.):
        v, po, p = self.v, self.po, self.p
        po_fresh = po.with_values(lambda x, y:  400.0*self.p_a+ po.values.mean*(1-self.p_a)*(x>=0))
        po = field.where(self.hvac_pos, po_fresh, po)
        po_in = self.p_alpha_C * self.Pocoeff[self.t] * resample(Box(x=(2.5, 3.1), y = (0.7,0.9)), to=po, soft=True)
        po = advect.mac_cormack(po, v, dt) + po_in
        po = diffuse.explicit(po, self.p_k_C, dt, substeps=3)
        if self.t % self.sT == self.sT-1:
         po = math.stop_gradient(po)
        self.pos.append(po.values.native('x,y').detach().numpy())
        return po

    def step_temperature(self, dt=1.):
        v, temp, p = self.v, self.temp, self.p
        temp_in = self.p_alpha_T * self.Pocoeff[self.t] * resample(Box(x=(2.5, 3.1), y = (0.7,0.9)), to=temp, soft=True) 
        temp = advect.mac_cormack(temp, v, dt) + temp_in
        temp = diffuse.explicit(temp, self.p_k_T, dt, substeps=4)
        temp_fresh = temp.with_values(lambda x, y:  self.u_airt[self.t] * (x>=0))
        temp = field.where(self.hvac_pos, temp_fresh, temp)
        temp_outside = temp.with_values(lambda x, y:  self.temp_out[self.t] * (x>=0))
        temp = field.where(self.outside, temp_outside, temp)
        #if self.t % 5 == 4:
        #  temp = math.stop_gradient(temp)
        self.temps.append(temp.values.native('x,y').detach().numpy())
        return temp

    def step(self, dt=1.): # 1 minute
        if self.simulate_c:
          po = self.step_co2(dt)
          self.po = po
        if self.simulate_t:
          temp = self.step_temperature(dt)
          self.temp = temp
        # velocity - apply boundary
        v, p = self.v, self.p
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, self.p_v, dt, substeps=4)
        v, p = fluid.make_incompressible(v, (self.ob1, self.ob2, self.ob3), Solve(x0=p))
        airflow = v.with_values(lambda x, y: tensor(vec(x=0, y= - self.p_u[self.t] * self.M *(x<=3.3)*(x>=2.7))))
        v = field.where(self.airflow_pos, airflow, v)
        if self.t % self.sT == self.sT-1:
          self.v = math.stop_gradient(v)
          self.p = math.stop_gradient(p)
        else:
          self.v = v
          self.p = p
        if self.plot_flag: self.update_plot()
        self.t += 1
        if self.simulate_c and self.simulate_t:
          return v, po, temp
        elif self.simulate_c: return v, po
        elif self.simulate_t: return v, temp

    def init_sys(self, c0=450, t0=21.):
        v_value = math.tensor(np.load(f'data/velocity/{self.v_df}.npz')['v'], spatial('x,y'), channel(vector='x,y'))
        self.v =  StaggeredGrid(v_value, extrapolation.combine_sides(x=(0., 0.), y=(0., ZERO_GRADIENT)), x=42, y=27, bounds=Box(x=4.2, y=2.7))
        if self.simulate_c:
          if self.po_df: po_value = math.tensor(np.load(f'data/velocity/{self.po_df}.npz')['po'], spatial('x,y'))
          else: po_value = c0
          self.po = CenteredGrid(po_value, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
          self.hvac_pos = self.po.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
        if self.simulate_t:
          np.random.seed(0)
          t0 = np.random.uniform(21.0, 21.5)
          self.temp = CenteredGrid(t0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
          self.hvac_pos = self.temp.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
          self.outside = self.temp.with_values(lambda x, y: (x>=4.1)*(y<=2.7)*(y>=0))
        self.airflow_pos = self.v.with_values(lambda x, y: vec(x=0, y=y>=2.65))
        self.ob1 = Obstacle(Box(x=(1.5, 2.7), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,)) 
        self.ob2 = Obstacle(Box(x=(0, 0.9), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.ob3 = Obstacle(Box(x=(3.3, 4.2), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.p = None
        self.Pocoeff = [0] * 5 + [2.0] * 20 + [0] * 10 + [3.0] * 20 + [0] * 5
        self.temp_out = 21.0 * np.ones(60)
        self.t = 0



