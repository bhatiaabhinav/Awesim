# import gymnasium as gym
# import numpy as np

# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Pendulum-v1", render_mode="rgb_array")

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     env.render("human")

import os
import sys
import gymnasium as gym

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gymenv import AwesimEnv
from bindings import *

env = AwesimEnv(city_width=1000, num_cars=256, decision_interval=0.1, sim_duration=60 * 5, synchronized=False, synchronized_sim_speedup=200.0, should_render=False)

load_path = sys.argv[1] if len(sys.argv) > 1 else None
if load_path is not None and not os.path.exists(load_path):
    print(f"Model not found at {load_path}")
    sys.exit(1)

if load_path is not None:
    print(f"Loading model from {load_path}")
    model = DDPG.load(load_path, env=env)
else:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64], n_critics=2), learning_starts=1000, batch_size=64, train_freq=1, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100_000, log_interval=4)
    model.save("ddpg_awesim")

env.should_render = True
env.synchronized = True
env.synchronized_sim_speedup = 1.0

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # VecEnv resets automatically when done
