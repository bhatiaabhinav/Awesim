import os
import sys
import gymnasium as gym

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from gymenv import AwesimEnv
from bindings import *

env = AwesimEnv(city_width=1000, num_cars=256, decision_interval=0.1, sim_duration=60 * 5, synchronized=False, synchronized_sim_speedup=200.0, should_render=False)

load_path = sys.argv[1] if len(sys.argv) > 1 else None
if load_path is not None and not os.path.exists(load_path):
    print(f"Model not found at {load_path}")
    sys.exit(1)

if load_path is not None:
    print(f"Loading model from {load_path}")
    model = SAC.load(load_path, env=env)
else:
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[128, 128]), learning_starts=1000, batch_size=256, learning_rate=0.0001, train_freq=1, verbose=1, device="cpu", tensorboard_log="./logs/sac_awesim_tensorboard/")

    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="sac_awesim",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=1000_000, log_interval=4, callback=checkpoint_callback, progress_bar=True, tb_log_name="demo_run", reset_num_timesteps=True)
    model.save("sac_awesim")

env.should_render = True
env.synchronized = True
env.synchronized_sim_speedup = 1.0

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # VecEnv resets automatically when done
