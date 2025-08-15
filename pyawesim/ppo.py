import os
import sys
import gymnasium as gym

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from gymenv import AwesimEnv
from bindings import *

experiment_name = "newdas_ppo_N8M1024_mb64_lr0.0001_ent0.01_kl0.02_gae0.95_AEB0.5_ctrl_speed_merge_rew_displacement_minus_heatloss_128-64-32"

def make_env(i=0):
    return AwesimEnv(city_width=1000, num_cars=256, decision_interval=1, sim_duration=60 * 5, i=i)

if __name__ == "__main__":

    n_totalsteps_per_batch = 8192
    n_envs = 8
    n_steps = n_totalsteps_per_batch // n_envs
    mb_size = 64

    load_path = sys.argv[1] if len(sys.argv) > 1 else None
    if load_path is not None and not os.path.exists(load_path):
        print(f"Model not found at {load_path}")
        sys.exit(1)

    if load_path is not None:
        print(f"Loading model from {load_path}")
        model = PPO.load(load_path)
    else:
        vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", vec_env, policy_kwargs=dict(net_arch=[128, 64, 32]), n_steps=n_steps, learning_rate=0.0001, ent_coef=0.01, batch_size=mb_size, target_kl=0.02, gae_lambda=0.95, verbose=1, device="cpu", tensorboard_log="./logs/awesim_tensorboard/")

        # Save a checkpoint every 10000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10000 // n_envs, 1),
            save_path="./models/"+experiment_name+ "/",
            name_prefix=experiment_name,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        n_ppo_iters = 10000
        model.learn(total_timesteps=n_ppo_iters * n_totalsteps_per_batch, log_interval=1, callback=checkpoint_callback, progress_bar=True, tb_log_name=experiment_name, reset_num_timesteps=True)
        model.save("models/" + experiment_name)

    env = make_env()
    env.should_render = True
    env.synchronized = True
    env.synchronized_sim_speedup = 16.0

    vec_env = make_vec_env(lambda: env)
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # VecEnv resets automatically when done
