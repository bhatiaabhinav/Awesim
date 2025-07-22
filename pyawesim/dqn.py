# import gymnasium as gym

# from stable_baselines3 import DQN

# env = gym.make("CartPole-v1", render_mode="human")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

import gymnasium as gym

# from stable_baselines3 import DQN
from sb3_ddqn_impl import DoubleDQN

from gymenv import AwesimEnv
from bindings import *

env = AwesimEnv(city_width=1000, num_cars=256, decision_interval=0.1, sim_duration=60 * 5, synchronized=False, synchronized_sim_speedup=200.0, should_render=False)

model = DoubleDQN("MlpPolicy", env, learning_starts=1000, batch_size=128, exploration_final_eps=0.1, train_freq=1, target_update_interval=2000, tau=1, verbose=1)
model.learn(total_timesteps=100_000, log_interval=4)

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