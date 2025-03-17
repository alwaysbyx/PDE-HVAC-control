# PDE-HVAC-control

This repository contains the code and data to reproduce the results in our work: **Ventilation and Temperature Control for Energy-efficient and Healthy Buildings: A Differentiable PDE Approach**, which has been accepted by Applied Energy.  You can get access to it via this [link](https://www.sciencedirect.com/science/article/abs/pii/S0306261924008602).
![Our framework](./image/figure1.png)


# Installation  

You are required to install 
- phiflow https://github.com/tum-pbs/PhiFlow
- gymnasium 
- stable_baselines3 https://stable-baselines3.readthedocs.io/en/master/guide/install.html

# Learning for PDEs in smart buildings
```python
cd learning
python learn_real.py # for real experiment
python learn_synthetic.py # for synthetic experiment
```

# Controlling PDEs in smart buildings
```python
cd control
python train_diffpde.py # our approach
python train_rl.py # reinforcement learning 
ODE_control.ipynb -- ODE approach
```


# Citation 
If this is useful for your work, please cite our paper. 
