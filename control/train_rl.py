from stable_baselines3 import PPO
import gymnasium as gym
import RLenv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from tqdm import tqdm


params = {'v': 0.00108, 'k_C': 0.00108,  'k_T':0.002,  'alpha_T': 0.3,  'a': 0.65, 'alpha_C': 50., "v_df": "v0", "simulate":"ct"}

def train():
    params['save'] = False
    env = gym.make("BuildingEnv-v0", params=params)
    model = PPO("MlpPolicy", env, n_steps=360*4, verbose=1, seed=0, tensorboard_log="./ppo_tensorboard/")
    checkpoint = CheckpointCallback(
        save_freq = 5000,
        save_path = "data/RL",
        name_prefix = "your_model",
        save_replay_buffer = True,
        save_vecnormalize = True
    )
    model.learn(total_timesteps=40000, log_interval=1, callback=checkpoint)


Test_Pocoeff = [[0] * 10 + [3.0] * 20 + [0] * 20 + [3.0] * 20 + [0] * 40, \
                [0] * 10 + [6.0] * 20 + [0] * 10 + [6.0] * 20 + [0] * 20, \
                [0] * 10 + [5.0] * 20 + [0] * 20 + [5.0] * 25 + [0] * 35, \
                [0] * 10 + [5.0] * 20 + [0] * 20 + [2.0] * 25 + [0] * 35, \
                [0] * 10 + [5.0] * 20 + [0] * 20 + [2.0] * 25 + [0] * 35, \
                [0] * 10 + [5.0] * 20 + [0] * 20 + [2.0] * 25 + [0] * 35]


def evalute():
    params['save'] = True
    env = gym.make("BuildingEnv-v0", params=params)
    model = PPO.load("your_model") # load your models
    obs, _ = env.reset()
    #env.Pocoeff = Test_Pocoeff[1]
    U, UT = [], []
    for i in tqdm(range(360)):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        U.append(action[0])
        UT.append(action[1])
        obs, rewards, dones, _, info = env.step(action)
    np.savez(f'data/baselines/rl.npz',  po=np.array(env.pos), temp = np.array(env.temps), occu=np.array(env.Pocoeff), temp_out = env.temp_out, u=np.array(U), u_air_t = np.array(UT))

if __name__ == "__main__":
    train()
    #evalute()