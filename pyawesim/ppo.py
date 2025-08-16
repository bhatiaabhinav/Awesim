import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from gymenv import AwesimEnv
# from bindings import *  # noqa


experiment_name = "ppo"
notes = "PPO with speed/merge/turn adjustment and AEB."

config = dict(
    experiment_name=experiment_name,
    n_totalsteps_per_batch=8192,
    ppo_iters=10000,
    normalize=True,
)
env_config = dict(
    city_width=1000,
    num_cars=256,
    decision_interval=1,
    sim_duration=60 * 5
)
vec_env_config = dict(n_envs=16)
ppo_config = dict(
    n_steps=config["n_totalsteps_per_batch"] // vec_env_config["n_envs"],
    learning_rate=0.0001,
    ent_coef=0.01,
    batch_size=64,
    target_kl=0.02,
    gae_lambda=0.95,
    verbose=1,
    device="cpu",
    tensorboard_log="./logs/awesim_tensorboard/"
)
policy_config = dict(
    net_arch=[512, 256, 128],
)
config = dict(**config, **env_config, **vec_env_config, **ppo_config, **policy_config)


def make_env(i=0):
    return AwesimEnv(**env_config, i=i)


if __name__ == "__main__":

    eval_mode = len(sys.argv) > 1

    if not eval_mode:
        vec_env = make_vec_env(make_env, **vec_env_config, vec_env_cls=SubprocVecEnv)
        if config["normalize"]:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, training=True)
        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_config, **ppo_config)
        wandb = wandb.init(project="awesim", name=experiment_name, config=config, notes=notes, sync_tensorboard=True, save_code=True)

        # Save a checkpoint every 50000 steps
        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=max(100000 // vec_env_config["n_envs"], 1),
                save_path="./models/" + experiment_name + "/",
                name_prefix=experiment_name,
                save_replay_buffer=False,
                save_vecnormalize=True,
            ),
            WandbCallback(),
        ])

        n_ppo_iters = 10000
        model.learn(total_timesteps=n_ppo_iters * config["n_totalsteps_per_batch"], log_interval=1, callback=callbacks, progress_bar=True, tb_log_name=experiment_name, reset_num_timesteps=True)
        model_path = "models/" + experiment_name + "/model.zip"
        vec_stats_path = "models/" + experiment_name + "/vec_normalize.pkl"
        model.save(model_path)
        if config["normalize"]:
            vec_env.save(vec_stats_path)  # type: ignore
    else:
        load_model_step = int(sys.argv[1])
        model_path = f"models/{experiment_name}/{experiment_name}_{load_model_step}_steps.zip"
        vec_stats_path = f"models/{experiment_name}/{experiment_name}_vecnormalize_{load_model_step}_steps.pkl"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            sys.exit(1)
        if config["normalize"] and not os.path.exists(vec_stats_path):
            print(f"Vec normalization stats not found at {vec_stats_path}")
            sys.exit(1)

    env = make_env()
    env.should_render = True
    env.synchronized = True
    env.synchronized_sim_speedup = 16.0
    env.render_server_ip = "127.0.0.1"

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device=config["device"])
    vec_env = make_vec_env(lambda: env)
    if config["normalize"]:
        print(f"Loading VecNormalize stats from {vec_stats_path}")
        vec_env = VecNormalize.load(vec_stats_path, vec_env)
        vec_env.training = False  # Disable training mode for evaluation
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = vec_env.step(action)
        # VecEnv resets automatically when done
