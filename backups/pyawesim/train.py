from typing import Union

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from gymenv import AwesimEnv
from gymenv_utils import EntropyDecayCallback, POLICY_NET_1024_512_256_RELU_LN_CONFIG, POLICY_NET_1024_512_256_RELU_LN_SQUASH_OUTPUT_CONFIG
from squashed_policy import SquashedActorCriticPolicy


# Experiment metadata
EXPERIMENT_NAME = "SquashOutput_DirectSpeedControl_DirectTHWControl_PPO_Lane84_ADAS_3Actions_LayerNorm_Gamma0.999"
NOTES = """Fixes: squash output. Trains a PPO policy in a gym environment where an agent controls a vehicle using ADAS with 3 actionable parameters, updated every second: cruise speed = action[0] * top_speed (with 0.9 smoothing step size from old target), turn/safe-merge signal, and follow distance = action[2] * 5s (with 0.9 smoothing step size from old value). Automatic Emergency Braking is always active with fixed parameters (2 mph, 0.01 mph, 3 ft, 6 ft; see ai.h). The goal is to reach lane #84 (2nd Ave N, between 1st St and S-2) from a random start without crashing, with episodes truncated after 10 minutes. Reward: -0.2/s, +100 for goal, -100 for crash (terminates episode). Observation includes 149 variables: vehicle state/capabilities (varies per episode), ADAS state, map awareness, nearby vehicles, intersection state, road/environment factors. Key hyperparameters: LayerNorm after each hidden layer in a [149, 1024, 512, 256, value/policy] architecture, gamma=0.999, normalize obs/rewards."""

# Configuration dictionaries
CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    # "n_totalsteps_per_batch": 32768,
    "n_totalsteps_per_batch": 8192, # For smaller machines
    "ppo_iters": 10000,
    "normalize": True,
    "norm_obs": True,
    "norm_reward": True,
}
ENV_CONFIG = {
    "goal_lane": 84,
    "city_width": 1000,
    "num_cars": 256,
    "decision_interval": 1,
    "sim_duration": 60 * 10,
    "goal_reward": 100.0,
    "crash_penalty": 100.0,
    "time_penalty_per_second": 0.2,
}
# VEC_ENV_CONFIG = {"n_envs": 64}
VEC_ENV_CONFIG = {"n_envs": 16}   # For smaller machines
PPO_CONFIG = {
    "n_steps": CONFIG["n_totalsteps_per_batch"] // VEC_ENV_CONFIG["n_envs"],
    "learning_rate": 0.0001,
    "ent_coef": 0.0,
    # "batch_size": 256,
    "batch_size": 64, # For smaller machines
    "target_kl": 0.02,
    "gae_lambda": 0.95,
    "gamma": 0.999,
    "verbose": 1,
    "device": "cuda",
    "tensorboard_log": "./logs/awesim_tensorboard/",
    "use_sde": False,
}
POLICY_CLS = SquashedActorCriticPolicy
POLICY_CONFIG = POLICY_NET_1024_512_256_RELU_LN_SQUASH_OUTPUT_CONFIG

# Combined configuration for logging
FULL_CONFIG = {**CONFIG, **ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def setup_training_env() -> Union[VecNormalize, VecEnv]:
    """Set up vectorized environment with normalization."""
    vec_env = make_vec_env(make_env, **VEC_ENV_CONFIG, vec_env_cls=SubprocVecEnv) # type: ignore
    if CONFIG["normalize"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=CONFIG["norm_obs"],
            norm_reward=CONFIG["norm_reward"],
            training=True,
            gamma=PPO_CONFIG["gamma"],
        )
    return vec_env


def setup_callbacks() -> CallbackList:
    """Configure training callbacks for checkpointing and logging."""
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000 // VEC_ENV_CONFIG["n_envs"], 1),
        save_path=f"./models/{EXPERIMENT_NAME}/",
        name_prefix=EXPERIMENT_NAME,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    entropy_callback = EntropyDecayCallback(start=0.01, end=PPO_CONFIG["ent_coef"], end_fraction=0.9)
    wandb_callback = WandbCallback()
    return CallbackList([entropy_callback, checkpoint_callback, wandb_callback])


def train_model() -> tuple[PPO, Union[VecNormalize, VecEnv]]:
    """Train the PPO model with the specified configuration."""
    vec_env = setup_training_env()
    model = PPO(POLICY_CLS, vec_env, policy_kwargs=POLICY_CONFIG, **PPO_CONFIG)

    project = "awesim" if ENV_CONFIG["goal_lane"] is None else "awesim_goal"
    wandb.init(
        project=project,
        name=EXPERIMENT_NAME,
        config=FULL_CONFIG,
        notes=NOTES,
        sync_tensorboard=True,
        save_code=True,
    )
    
    model.learn(
        total_timesteps=CONFIG["ppo_iters"] * CONFIG["n_totalsteps_per_batch"],
        log_interval=1,
        callback=setup_callbacks(),
        progress_bar=True,
        tb_log_name=EXPERIMENT_NAME,
        reset_num_timesteps=True,
    )
    
    model_path = f"models/{EXPERIMENT_NAME}/model.zip"
    vec_stats_path = f"models/{EXPERIMENT_NAME}/vec_normalize.pkl"
    model.save(model_path)
    if isinstance(vec_env, VecNormalize):
        vec_env.save(vec_stats_path)
    
    return model, vec_env

if __name__ == "__main__":
    train_model()