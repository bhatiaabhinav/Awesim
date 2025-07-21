# import gymnasium as gym

# from stable_baselines3 import A2C

# env = gym.make("CartPole-v1", render_mode="rgb_array")

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     # if done:
#     #   obs = vec_env.reset()

import gymnasium as gym

from stable_baselines3 import A2C

# env = gym.make("CartPole-v1", render_mode="rgb_array")

from gymenv import AwesimEnv
from bindings import *

env = AwesimEnv(city_width=1000, num_cars=256, decision_interval=0.1, sim_duration=60 * 5, synchronized=False, synchronized_sim_speedup=100.0, should_render=True)


# env.reset()
# env.reset()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

env.should_render = True
env.synchronized = True
env.synchronized_sim_speedup = 1.0

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()