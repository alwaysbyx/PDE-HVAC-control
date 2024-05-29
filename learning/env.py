from phi.torch.flow import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from phiml.math import reshaped_numpy
import torch
import numpy as np

class EnvPDE:
    def __init__(self, params, simulator='c', plot=False):
        self.simulate_t = 't' in simulator
        self.simulate_c = 'c' in simulator
        self.p_v = torch.tensor(params['v'], requires_grad=True)
        self.p_a = torch.tensor(params['a'], requires_grad=True)
        self.p_u = torch.tensor(params['u'], requires_grad=True)
        if self.simulate_c:
            self.p_alpha_C = torch.tensor(params['alpha_C'], requires_grad=True)
            self.p_k_C = torch.tensor(params['k_C'], requires_grad=True)
        if self.simulate_t:
            self.p_alpha_T = torch.tensor(params['alpha_T'], requires_grad=True)
            self.p_k_T = torch.tensor(params['k_T'], requires_grad=True)
        self.init_plot(plot)
        self.init_sys()
        self.pos = []

    def init_plot(self, plot):
        self.plot_flag = plot
        if plot:
            self.rows, self.cols = [28, 10, 2], [-1, -1, 15]
            self.fig, self.ax = plt.subplots(figsize=(8,6))
    
    def update_plot(self):
        data = self.v.at_centers()
        vector = data.bounds.shape['vector']
        extra_channels = data.shape.channel.without('vector')
        x, y = reshaped_numpy(data.points.vector[0,1], [vector, data.shape.without('vector')])
        u, v = reshaped_numpy(data.values.vector[0,1], [vector, extra_channels, data.shape.without('vector')])
        for ch in range(u.shape[0]):
            self.ax.quiver(x, y,  u[ch], v[ch], color="black")
        po = self.po.values.native('y,x').detach().numpy() # [::-1, :]
        self.pos.append(po)
        x = np.linspace(0, 4.2, 42)
        y = np.linspace(0, 2.7, 27)
        X, Y = np.meshgrid(x,y)
        self.ax.imshow(po[::-1], extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap, interpolation_stage='rgba')
        #im = self.ax.imshow(po, origin='lower', vmin=400., vmax=2000.)
        #self.cb = plt.colorbar(im, ax=self.ax)
        plt.savefig(f'image/{self.t}.png', dpi=300)
        plt.pause(1e-3)
        #self.cb.remove()
        self.ax.clear()
    
    def save(self, name='train'):
        if self.t % 10 == 9:
            np.savez(f'data/simulate_{name}.npz', c=np.array(self.pos))

    def step(self, dt=1.): # 1 minute
        v, po, p = self.v, self.po, self.p
        # Po - apply boundary
        po_in = self.p_alpha_C * self.Pocoeff[self.t] * resample(Box(x=(2.5,3.1), y = (0.7,0.9)), to=po, soft=True)
        po = advect.mac_cormack(po, v, dt) + po_in
        po = diffuse.explicit(po, self.p_k_C, dt, substeps=6)
        po_fresh = po.with_values(lambda x, y:  400.0*self.p_a+ po.values.x[9:15].y[25:].mean*(1-self.p_a)*(x>=0))
        po = field.where(self.pollutant_pos, po_fresh, po)
        # velocity - apply boundary
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, self.p_v, dt, substeps=3)
        v, p = fluid.make_incompressible(v, (self.ob1, self.ob2), Solve(x0=p))
        airflow = v.with_values(lambda x, y: tensor(vec(x=0, y= - self.p_u * self.u[self.t] *(x<=3.3)*(x>=2.7))))
        v = field.where(self.airflow_pos, airflow, v)
        if self.t % 5 == 4:
            self.v = math.stop_gradient(v)
            self.po = math.stop_gradient(po)
            self.p = math.stop_gradient(p)
        else:
            self.v, self.po, self.p = v, po, p
        if self.plot_flag: self.update_plot()
        self.t += 1
        self.pos.append(po.values.native('x,y').detach().numpy())
        return v, po, p

    def init_sys(self, c0 = 400.):
        self.v =  StaggeredGrid(0, extrapolation.combine_sides(x=(0., 0.), y=(0., ZERO_GRADIENT)), x=42, y=27, bounds=Box(x=4.2, y=2.7))
        self.po = CenteredGrid(c0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
        self.airflow_pos = self.v.with_values(lambda x, y: vec(x=0, y=y>=2.65))
        self.pollutant_pos = self.po.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
        self.ob1 = Obstacle(Box(x=(1.5, 2.8), y=(2.65, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.ob2 = Obstacle(Box(x=(0, 0.9), y=(2.65, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.p = None
        self.Pocoeff = [0] * 20 + ([1.0] * 30 + [0] * 30) * 2  #+ [0] * int(2.5*60)
        self.u = [1.0] * len(self.Pocoeff)
        self.t = 0



class EnvPDE2:
    def __init__(self, params, simulator='c', plot=False):
        self.simulate_t = 't' in simulator
        self.simulate_c = 'c' in simulator
        self.p_v = torch.tensor(params['v'], requires_grad=True)
        self.p_a = torch.tensor(params['a'], requires_grad=True)
        self.p_u = torch.tensor(params['u'], requires_grad=True)
        if self.simulate_c:
            self.p_alpha_C = torch.tensor(params['alpha_C'], requires_grad=True)
            self.p_k_C = torch.tensor(params['k_C'], requires_grad=True)
        if self.simulate_t:
            self.p_alpha_T = torch.tensor(params['alpha_T'], requires_grad=True)
            self.p_k_T = torch.tensor(params['k_T'], requires_grad=True)
        self.init_plot(plot)
        self.init_sys()
        self.pos = []
        self.temps = []

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
        po_fresh = po.with_values(lambda x, y:  400.0*self.p_a+ po.values.mean*(1-self.p_a)*(x>=0)) #TODO
        po = field.where(self.hvac_pos, po_fresh, po)
        po_in = self.p_alpha_C * self.Pocoeff[self.t] * resample(Box(x=(2.5, 3.1), y = (0.7,0.9)), to=po, soft=True)
        po = advect.mac_cormack(po, v, dt) + po_in
        po = diffuse.explicit(po, self.p_k_C, dt, substeps=4)
        if self.t % 5 == 4:
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
        if self.t % 5 == 4:
          temp = math.stop_gradient(temp)
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
        airflow = v.with_values(lambda x, y: tensor(vec(x=0, y= - self.p_u * self.u[self.t] *(x<=3.3)*(x>=2.7))))
        v = field.where(self.airflow_pos, airflow, v)
        if self.t % 5 == 4:
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

    def init_sys(self, c0=400, t0=21.):
        self.v =  StaggeredGrid(0, extrapolation.combine_sides(x=(0., 0.), y=(0., ZERO_GRADIENT)), x=42, y=27, bounds=Box(x=4.2, y=2.7))
        if self.simulate_c:
          self.po = CenteredGrid(c0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
          self.hvac_pos = self.po.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
        if self.simulate_t:
          self.temp = CenteredGrid(t0, extrapolation.combine_sides(x=(ZERO_GRADIENT, ZERO_GRADIENT), y = (ZERO_GRADIENT, ZERO_GRADIENT)),  x=42, y=27, bounds=Box(x=4.2, y=2.7))
          self.hvac_pos = self.temp.with_values(lambda x, y: (y>=2.6)*(x<=3.3)*(x>=2.7))
        self.airflow_pos = self.v.with_values(lambda x, y: vec(x=0, y=y>=2.65))
        self.ob1 = Obstacle(Box(x=(1.5, 2.7), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,)) #TODO
        self.ob2 = Obstacle(Box(x=(0, 0.9), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.ob3 = Obstacle(Box(x=(3.3, 4.2), y=(2.69, 2.7)), velocity=[0., 0], angular_velocity=tensor(0,))
        self.p = None
        self.M = 1.2                # maximum airflow rate
        self.u = [1.0] * 210        # flow rate
        self.u_airt = [21.0] * 210  # flow rate
        self.Pocoeff = [0] * 5 + [2.0] * 20 + [0] * 10 + [3.0] * 20 + [0] * 5
        self.t = 0



if __name__ == "__main__":
    params = {'v': 2e-3, 'k_C': 1e-3, 'a': 0.648, 'alpha_C': 750., 'u': 1.229}
    env = EnvPDE(params, plot=True)
    po_record = []
    for t in tqdm(range(180)):
        v, po, _ = env.step(dt=1.)
        po_record.append(po.values.numpy('x,y'))
        np.savez('data/simulate_co2.npz', C=np.array(po_record))
        #np.savez('data/warm_velocity.npz', v=v_array)
    # po_record_ = np.array(po_record)
    # plt.plot(po_record_[:,28, -1], label='supply')
    # plt.plot(po_record_[:,5, 15], label='blackboard')
    # plt.plot(po_record_[:,17, -1], label='return')
    # plt.legend()
    # plt.savefig('data/simulate.png', dpi=300)
    # plt.show()

