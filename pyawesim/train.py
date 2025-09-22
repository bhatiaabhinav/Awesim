import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from gymenv import AwesimEnv
from gymenv_utils import make_mlp
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv
from ppo import Actor, Critic, PPO


# Experiment metadata
EXPERIMENT_NAME = "ApptWorld-20mins"
NOTES = """From a random starting point, reach destination (lane #84) in 20 mins (rew = 1, 0 otherwise), else lost wage cost starts accumulating (in a seperate cost stream). Cost of gas and cost of crash also included in the cost stream. Episode ends at 60 mins and wage for entire 8-hour workday is lost if not reached. No CMDP solution implemented yet."""

# Configuration dictionaries
CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "n_totalsteps_per_batch": 32768,    # total steps across all parallel envs per PPO update
    "ppo_iters": 10000,                 # number of PPO updates
    "norm_obs": True,                   # whether to normalize observations by running estimates of mean and stddev
    "norm_reward": True,                # whether to normalize rewards by running estimates of stddev of discounted returns
}
ENV_CONFIG = {
    "goal_lane": 84,
    "city_width": 1000,                 # in meters
    "num_cars": 256,                    # number of other cars in the environment
    "decision_interval": 1,             # in seconds, how often the agent can take an action (reconfigure the driving assist)
    "sim_duration": 60 * 60,            # 60 minutes max duration after which the episode ends and wage for entire 8-hour workday is lost
    "appointment_time": 60 * 20,        # 20 minutes to reach the appointment
    "cost_gas_per_gallon": 4.0,         # in NYC (USD)
    "cost_car": 30000,                  # cost of the car (USD), incurred if crashed (irrespective of severity)
    "cost_lost_wage_per_hour": 40,      # of a decent job in NYC (USD)
    "init_fuel_gallons": 3.0,           # initial fuel in gallons -- 1/4th of a 12-gallon tank car
    "goal_reward": 1.0,                 # A small positive number to incentivize reaching the goal
}
VEC_ENV_CONFIG = {"n_envs": 64}
PPO_CONFIG = {
    "iters": CONFIG["ppo_iters"],
    "nsteps": CONFIG["n_totalsteps_per_batch"] // VEC_ENV_CONFIG["n_envs"],
    "actor_lr": 0.0001,
    "critic_lr": 0.0001,
    "decay_lr": False,
    "entropy_coef": 0.0,
    "batch_size": 256,
    "target_kl": 0.01,
    "lam": 0.95,
    "gamma": 0.999,
    "clipnorm": 0.5,
    "norm_rewards": CONFIG["norm_reward"],
    # "verbose": 1,
    "training_device": "cuda" if torch.cuda.is_available() else "cpu",
    "inference_device": "cuda" if torch.cuda.is_available() else "cpu",
}
POLICY_CONFIG = {
    "net_arch": [1024, 512, 256],
    "layernorm": True,
}
# Combined configuration for logging
FULL_CONFIG = {**CONFIG, **ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def setup_training_env() -> VectorEnv:
    return AsyncVectorEnv([lambda: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)


def train_model(no_wandb=False) -> Tuple[Actor, Critic, PPO]:
    """Train the PPO model with the specified configuration."""

    vec_env = setup_training_env()

    project = "Awesim-ApppointmentWorld"
    if not no_wandb:
        wandb.init(
            project=project,
            name=EXPERIMENT_NAME,
            config=FULL_CONFIG,
            notes=NOTES,
            save_code=True,
        )

    obs_dim = vec_env.single_observation_space.shape[0]   # type: ignore
    action_dim = vec_env.single_action_space.shape[0]     # type: ignore
    print(f"Observation space dim: {obs_dim}, Action space dim: {action_dim}")

    actor_model = make_mlp(obs_dim, POLICY_CONFIG["net_arch"], action_dim, layernorm=POLICY_CONFIG["layernorm"])
    critic_model = make_mlp(obs_dim, POLICY_CONFIG["net_arch"], 1, layernorm=POLICY_CONFIG["layernorm"])
    actor = Actor(actor_model, vec_env.single_action_space, norm_obs=CONFIG["norm_obs"])    # type: ignore
    critic = Critic(critic_model, norm_obs=CONFIG["norm_obs"])

    ppo = PPO(vec_env, actor, critic, **PPO_CONFIG, custom_info_keys_to_log_at_episode_end=['crashed', 'reached_goal', 'reached_late', 'out_of_fuel', 'timeout', 'total_fuel_consumed'])

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
                "rollout/ep_cost_mean": ppo.stats['mean_total_costs'][-1],
                "rollout/success_rate": ppo.stats['mean_successes'][-1],
                "rollout/crash_rate": ppo.stats['info_key_means']['crashed'][-1],
                "rollout/reached_goal_rate": ppo.stats['info_key_means']['reached_goal'][-1],
                "rollout/out_of_fuel_rate": ppo.stats['info_key_means']['out_of_fuel'][-1],
                "rollout/reached_late_rate": ppo.stats['info_key_means']['reached_late'][-1],
                "rollout/timeout_rate": ppo.stats['info_key_means']['timeout'][-1],
                "rollout/mean_fuel_consumed": ppo.stats['info_key_means']['total_fuel_consumed'][-1],
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
    print(ppo.stats['timestepss'])
    print(ppo.stats['mean_returns'])
    print(ppo.stats['mean_total_costs'])
    plt.plot(ppo.stats['timestepss'], ppo.stats['mean_returns'])
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Training Curve -- Mean Reward (100-ep moving window) vs Timesteps")
    plt.savefig(f"./models/{EXPERIMENT_NAME}/training_curve_returns.png")
    plt.clf()
    plt.plot(ppo.stats['timestepss'], ppo.stats['mean_total_costs'])
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Total Cost")
    plt.title("Training Curve -- Mean Total Cost (100-ep moving window) vs Timesteps")
    plt.savefig(f"./models/{EXPERIMENT_NAME}/training_curve_total_costs.png")
