import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from gymenv import AwesimEnv
from gymenv_utils import make_mlp_with_layernorm
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv
from ppo import Actor, Critic, PPO


# Experiment metadata
EXPERIMENT_NAME = "PPO-CustomImpl-BigNet"
NOTES = """Trains a PPO policy with a custom PPO implementation in a gym environment where an agent controls a vehicle using ADAS with 3 actionable parameters, updated every second: cruise speed = action[0] * top_speed (with 0.9 smoothing step size from old target), turn/safe-merge signal, and follow distance = action[2] * 5s (with 0.9 smoothing step size from old value). Automatic Emergency Braking is always active with fixed parameters (2 mph, 0.01 mph, 3 ft, 6 ft; see ai.h). The goal is to reach lane #84 (2nd Ave N, between 1st St and S-2) from a random start without crashing, with episodes truncated after 10 minutes. Reward: -0.2/s, +100 for goal, -100 for crash (terminates episode). Observation includes 149 variables: vehicle state/capabilities (varies per episode), ADAS state, map awareness, nearby vehicles, intersection state, road/environment factors. Key hyperparameters: LayerNorm after each hidden layer in a [149, 1024, 512, 256, value/policy] architecture, gamma=0.999, NO normalize obs/rewards."""

# Configuration dictionaries
CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "n_totalsteps_per_batch": 32768,
    # "n_totalsteps_per_batch": 1024, # For smaller machines
    "ppo_iters": 100,
    "normalize": False,  # TODO: not supported yet
    "norm_obs": False,
    "norm_reward": False,
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
VEC_ENV_CONFIG = {"n_envs": 64}
# VEC_ENV_CONFIG = {"n_envs": 8}   # For smaller machines
PPO_CONFIG = {
    "iters": CONFIG["ppo_iters"],
    "nsteps": CONFIG["n_totalsteps_per_batch"] // VEC_ENV_CONFIG["n_envs"],
    "actor_lr": 0.0001,
    "critic_lr": 0.0001,
    "decay_lr": False,
    "entropy_coef": 0.0,
    "batch_size": 256,
    # "batch_size": 64, # For smaller machines
    "target_kl": 0.01,
    "lam": 0.95,
    "gamma": 0.999,
    # "verbose": 1,
    "training_device": "cpu",
    "inference_device": "cpu",
}
POLICY_CONFIG = {
    "net_arch": [1024, 512, 256],
}
# Combined configuration for logging
FULL_CONFIG = {**CONFIG, **ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def setup_training_env() -> VectorEnv:
    return AsyncVectorEnv([lambda: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], autoreset_mode=AutoresetMode.SAME_STEP)


def train_model(no_wandb=False) -> Tuple[Actor, Critic, PPO]:
    """Train the PPO model with the specified configuration."""

    vec_env = setup_training_env()

    project = "awesim" if ENV_CONFIG["goal_lane"] is None else "awesim_goal"
    if not no_wandb:
        wandb.init(
            project=project,
            name=EXPERIMENT_NAME,
            config=FULL_CONFIG,
            notes=NOTES,
            sync_tensorboard=True,
            save_code=True,
        )

    obs_dim = vec_env.single_observation_space.shape[0]   # type: ignore
    action_dim = vec_env.single_action_space.shape[0]     # type: ignore

    actor_model = make_mlp_with_layernorm(obs_dim, POLICY_CONFIG["net_arch"], action_dim)
    critic_model = make_mlp_with_layernorm(obs_dim, POLICY_CONFIG["net_arch"], 1)
    actor = Actor(actor_model, vec_env.single_action_space)    # type: ignore
    critic = Critic(critic_model)

    ppo = PPO(vec_env, actor, critic, **PPO_CONFIG)

    # make dir for saving models
    model_dir = f"./models/{EXPERIMENT_NAME}"
    os.makedirs(model_dir, exist_ok=True)

    # interleave eval, logging, model_saving
    while ppo.stats['iterations'] < ppo.iters:
        ppo.train(1)
        if not no_wandb:
            wandb.log({
                "global_step": ppo.stats['total_timesteps'],
                "iteration": ppo.stats['iterations'],
                "episodes": ppo.stats['total_episodes'],
                "rollout/ep_rew_mean": ppo.stats['mean_returns'][-1],
                "rollout/ep_len_mean": ppo.stats['mean_lengths'][-1],
                "rollout/success_rate": ppo.stats['mean_successes'][-1],
                "train/loss": ppo.stats["losses"][-1],
                "train/value_loss": ppo.stats["critic_losses"][-1],
                "train/policy_gradient_loss": ppo.stats["actor_losses"][-1],
                "train/mean_value": ppo.stats["values"][-1],
                "train/entropy_loss": -ppo.stats["entropies"][-1],
                "train/approx_kl": ppo.stats["kl_divs"][-1],
                "train/std": np.exp(ppo.stats["logstds"][-1]),
                "train/logprob": ppo.stats["logprobs"][-1],
            })
        if ppo.stats['iterations'] % 10 == 0:
            torch.save(actor.state_dict(), f"{model_dir}/actor_iter{ppo.stats['iterations']}.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_iter{ppo.stats['iterations']}.pth")
            torch.save(actor.state_dict(), f"{model_dir}/actor_latest.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_latest.pth")

    if not no_wandb:
        wandb.finish()

    actor.cpu()
    critic.cpu()

    # save the models
    torch.save(actor.state_dict(), f"{model_dir}/actor_final.pth")
    torch.save(critic.state_dict(), f"{model_dir}/critic_final.pth")

    return actor, critic, ppo


if __name__ == "__main__":
    actor, critic, ppo = train_model(True)
    plt.plot(ppo.stats['mean_returns'])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward (over 100-episode moving window)")
    plt.title("Custom PPO on AwesimEnv")
    plt.show()
