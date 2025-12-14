import ast
import os
import sys
from typing import Tuple

import gymnasium
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from gymenv import AwesimEnv
from pyawesim.cnn import SimpleCNN, SimpleCNNWith3x3DownsamplingPairs, MinimalRLConv128
from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode, VectorEnv
from ppo import Actor, ActorBase, Critic, PPO


class ModelArchitecture(nn.Module):

    def __init__(self, hidden_dim_per_image, hidden_dim: int, dim_out: int, c_base: int = 32, num_images: int = 9, cnn_type="minimal"):
        net_name_to_class = {
            "simple": SimpleCNN,
            "minimal": MinimalRLConv128,
            "simplefancy": SimpleCNNWith3x3DownsamplingPairs,
        }
        CNN = net_name_to_class[cnn_type]
        super().__init__()
        self.hidden_dim_per_image = hidden_dim_per_image
        self.hidden_dim = hidden_dim
        self.num_images = num_images
        self.cnns = nn.ModuleList([CNN(hidden_dim_per_image, c_base=c_base) for _ in range(num_images)])
        self.hidden_fc = nn.Linear(num_images * hidden_dim_per_image, hidden_dim)
        self.relu = nn.ReLU()
        self.final_fc = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:4] == (3 * self.num_images, 128, 128), f"Expected input shape (3 * {self.num_images}, 128, 128), but got {x.shape}"
        cnn_outputs = torch.zeros((x.shape[0], self.num_images * self.hidden_dim_per_image), device=x.device)  # (B, num_images * hidden_dim)
        for i, cnn in enumerate(self.cnns):
            x_i = x[:, i * 3:(i + 1) * 3, :, :]         # (B, 3, H, W)
            y_i = cnn(x_i)                              # (B, hidden_dim)
            cnn_outputs[:, i * self.hidden_dim_per_image:(i + 1) * self.hidden_dim_per_image] = y_i     # (B, hidden_dim)
        x = self.hidden_fc(cnn_outputs)                 # (B, hidden_dim)
        x = self.relu(x)                                # (B, hidden_dim)
        x = self.final_fc(x)                            # (B, dim_out)
        return x


WANDB_CONFIG = {
    "no_wandb": False,
    "project": "Awesim-ApppointmentWorld",
    "experiment_name": "Awesim-Vision",
    "notes": """Moved to vision based environment""",
    "video_interval": 25,
}
ENV_CONFIG = {
    "cam_resolution": (128, 128),       # width, height
    "anti_aliasing": True,
    "goal_lane": 84,
    "city_width": 1000,                 # in meters
    "num_cars": 256,                    # number of other cars in the environment
    "decision_interval": 1,             # in seconds, how often the agent can take an action (reconfigure the driving assist)
    "sim_duration": 60 * 60,            # 60 minutes max duration after which the episode ends and wage for entire 8-hour workday is lost
    "appointment_time": 60 * 20,        # 20 minutes to reach the appointment
    "cost_gas_per_gallon": 4.0,         # in NYC (USD)
    "cost_car": 0,                      # cost of the car (USD), incurred if crashed (irrespective of severity)
    "cost_lost_wage_per_hour": 40,      # of a decent job in NYC (USD)
    "may_lose_entire_day_wage": False,  # lose entire 8-hour workday wage if not reached by the end of sim_duration
    "init_fuel_gallons": 3.0,           # initial fuel in gallons -- 1/4th of a 12-gallon tank car
    "goal_reward": 1.0,                 # A small positive number to incentivize reaching the goal
    "reward_shaping": False,
}
VEC_ENV_CONFIG = {
    "n_envs": 8,
    "async": True,
}
PPO_CONFIG = {
    "iters": 10000,
    "nsteps": 256,
    "actor_lr": 0.0001,
    "critic_lr": 0.0001,
    "decay_lr": False,
    "decay_entropy_coef": True,
    "entropy_coef": 0.01,
    "batch_size": 64,
    "nepochs": 4,
    "target_kl": 0.01,
    "lam": 0.95,
    "gamma": 0.999,
    "clipnorm": 0.5,
    "norm_rewards": True,
    "cmdp_mode": False,
    "constrain_undiscounted_cost": True,
    "cost_threshold": 5.0,  # max cost per episode
    "lagrange_lr": 0.01,
    "lagrange_max": 100.0,
    "training_device": "cuda" if torch.cuda.is_available() else "cpu",
    "inference_device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": False
}
ENV_CONFIG["reward_shaping_gamma"] = PPO_CONFIG["gamma"]
NET_CONFIG = {
    "hidden_dim_per_image": 128,
    "hidden_dim": 512,
    "c_base": 32,
    "cnn_type": "simple",
}
POLICY_CONFIG = {
    "norm_obs": True,
    "state_dependent_std": True,
}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def setup_training_env() -> VectorEnv:
    if VEC_ENV_CONFIG["async"]:
        return AsyncVectorEnv([lambda i=i: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        return SyncVectorEnv([lambda i=i: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)


def train_model() -> Tuple[ActorBase, Critic, PPO]:
    """Train the PPO model with the specified configuration."""

    vec_env = setup_training_env()

    if not WANDB_CONFIG["no_wandb"]:
        wandb.init(
            project=WANDB_CONFIG["project"],
            name=WANDB_CONFIG["experiment_name"],
            config={**ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG},
            notes=WANDB_CONFIG["notes"],
            save_code=True,
        )

    obs_space: Box = vec_env.single_observation_space       # type: ignore
    print(f"Observation space: {obs_space}")
    action_space: Box = vec_env.single_action_space         # type: ignore
    obs_dim = vec_env.single_observation_space.shape[0]     # type: ignore
    action_dim = vec_env.single_action_space.shape[0]       # type: ignore
    print(f"Observation space dim: {obs_dim}, Action space dim: {action_dim}")

    actor_dim_out = action_dim * 2 if POLICY_CONFIG["state_dependent_std"] else action_dim
    actor_model = ModelArchitecture(dim_out=actor_dim_out, **NET_CONFIG)
    print("Actor model:")
    print(actor_model)
    critic_model = ModelArchitecture(dim_out=1, **NET_CONFIG)

    actor = Actor(actor_model, obs_space, action_space, norm_obs=POLICY_CONFIG["norm_obs"], state_dependent_std=POLICY_CONFIG["state_dependent_std"])    # type: ignore
    critic = Critic(critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    if PPO_CONFIG["cmdp_mode"]:
        cost_critic_model = ModelArchitecture(dim_out=1, **NET_CONFIG)
        cost_critic = Critic(cost_critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    else:
        cost_critic = None

    ppo = PPO(vec_env, actor, critic, cost_critic=cost_critic, **PPO_CONFIG, custom_info_keys_to_log_at_episode_end=['crashed', 'reached_goal', 'reached_in_time', 'reached_but_late', 'out_of_fuel', 'timeout', 'total_fuel_consumed', 'crashed_forward', 'crashed_backward', 'progress_m_during_crash', 'progress_m_during_front_crash', 'progress_m_during_back_crash', 'crashed_on_intersection', 'aeb_in_progress_during_crash'])  # type: ignore

    # make dir for saving models
    model_dir = f"./models/{WANDB_CONFIG['experiment_name']}"
    os.makedirs(model_dir, exist_ok=True)

    # an env for video recording
    env = None
    video_dir = f"videos/{WANDB_CONFIG['experiment_name']}"
    if WANDB_CONFIG["video_interval"] is not None:
        env = make_env()
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="training")
        if not WANDB_CONFIG["no_wandb"]:
            wandb.save(f"{video_dir}/*.mp4")

    # interleave eval, logging, model_saving
    while ppo.stats['iterations'] < ppo.iters:
        ppo.train(1)
        if not WANDB_CONFIG["no_wandb"]:
            wandb.log({
                "global_step": ppo.stats['total_timesteps'],
                "iteration": ppo.stats['iterations'],
                "episodes": ppo.stats['total_episodes'],
                "hours_of_experience": ppo.stats['total_timesteps'] * ENV_CONFIG['decision_interval'] / 3600.0,
                "rollout/ep_rew_mean": ppo.stats['mean_returns'][-1],
                "rollout/ep_len_mean": ppo.stats['mean_lengths'][-1],
                "rollout/ep_cost_mean": ppo.stats['mean_total_costs'][-1],
                "rollout/ep_cost_mean_discounted": ppo.stats['mean_discounted_costs'][-1],
                "rollout/success_rate": ppo.stats['mean_successes'][-1],
                "rollout/crash_rate": ppo.stats['info_key_means']['crashed'][-1],
                "rollout/reached_goal_rate": ppo.stats['info_key_means']['reached_goal'][-1],
                "rollout/out_of_fuel_rate": ppo.stats['info_key_means']['out_of_fuel'][-1],
                "rollout/reached_in_time_rate": ppo.stats['info_key_means']['reached_in_time'][-1],
                "rollout/reached_but_late_rate": ppo.stats['info_key_means']['reached_but_late'][-1],
                "rollout/timeout_rate": ppo.stats['info_key_means']['timeout'][-1],
                "rollout/mean_fuel_consumed": ppo.stats['info_key_means']['total_fuel_consumed'][-1],
                "rollout/crashed_forward_rate": ppo.stats['info_key_means']['crashed_forward'][-1],
                "rollout/crashed_backward_rate": ppo.stats['info_key_means']['crashed_backward'][-1],
                "rollout/mean_progress_m_during_crash": ppo.stats['info_key_means']['progress_m_during_crash'][-1],
                "rollout/mean_progress_m_during_front_crash": ppo.stats['info_key_means']['progress_m_during_front_crash'][-1],
                "rollout/mean_progress_m_during_back_crash": ppo.stats['info_key_means']['progress_m_during_back_crash'][-1],
                "rollout/crashed_on_intersection_rate": ppo.stats['info_key_means']['crashed_on_intersection'][-1],
                "rollout/aeb_in_progress_during_crash_rate": ppo.stats['info_key_means']['aeb_in_progress_during_crash'][-1],
                "train/loss": ppo.stats["losses"][-1],
                "train/value_loss": ppo.stats["critic_losses"][-1],
                "train/cost_value_loss": ppo.stats["cost_losses"][-1],
                "train/policy_gradient_loss": ppo.stats["actor_losses"][-1],
                "train/mean_value": ppo.stats["values"][-1],
                "train/mean_cost_value": ppo.stats["cost_values"][-1],
                "train/lagrange": ppo.stats['lagranges'][-1],
                "train/entropy_loss": -ppo.stats["entropies"][-1],
                "train/approx_kl": ppo.stats["kl_divs"][-1],
                "train/logprob": ppo.stats["logprobs"][-1],
            })
        if ppo.stats['iterations'] % 100 == 0:
            torch.save(actor.state_dict(), f"{model_dir}/actor_iter{ppo.stats['iterations']}.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_iter{ppo.stats['iterations']}.pth")
            torch.save(actor.state_dict(), f"{model_dir}/actor_latest.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_latest.pth")

        if env is not None and ppo.stats['iterations'] % WANDB_CONFIG["video_interval"] == 0:
            actor.evaluate_policy(env, 1, False)    # for video recording
            if not WANDB_CONFIG["no_wandb"]:
                wandb.save(f"{video_dir}/*.mp4")

    if not WANDB_CONFIG["no_wandb"]:
        wandb.finish()

    actor.cpu()
    critic.cpu()

    # save the models
    torch.save(actor.state_dict(), f"{model_dir}/actor_final.pth")
    torch.save(critic.state_dict(), f"{model_dir}/critic_final.pth")

    return actor, critic, ppo


if __name__ == "__main__":

    # --- override defaults with command line arguments in format key=value ---
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.strip()
            # try to interpret the value as Python literal (int, float, list, dict, bool, None)
            try:
                v = ast.literal_eval(v)
            except Exception:
                v = v  # keep as string if not literals
            found_key = False
            if k in ENV_CONFIG:
                ENV_CONFIG[k] = v
                found_key = True
            if k in VEC_ENV_CONFIG:
                VEC_ENV_CONFIG[k] = v   # type: ignore
                found_key = True
            if k in PPO_CONFIG:
                PPO_CONFIG[k] = v
                found_key = True
            if k in NET_CONFIG:
                NET_CONFIG[k] = v
                found_key = True
            if k in POLICY_CONFIG:
                POLICY_CONFIG[k] = v    # type: ignore
                found_key = True
            if k in WANDB_CONFIG:
                WANDB_CONFIG[k] = v
                found_key = True
            if not found_key:
                # print in yellow color
                print(f"\033[93mWarning: key {k} not found in any config, ignoring the override.\033[0m")

    print("Final configuration:")
    CONFIG = {**WANDB_CONFIG, **ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG}
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    actor, critic, ppo = train_model()
    print(ppo.stats['timestepss'])
    print(ppo.stats['mean_returns'])
    print(ppo.stats['mean_total_costs'])
    plt.plot(ppo.stats['timestepss'], ppo.stats['mean_returns'])
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Training Curve -- Mean Reward (100-ep moving window) vs Timesteps")
    plt.savefig(f"./models/{WANDB_CONFIG['experiment_name']}/training_curve_returns.png")
    plt.clf()
    plt.plot(ppo.stats['timestepss'], ppo.stats['mean_total_costs'])
    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Total Cost")
    plt.title("Training Curve -- Mean Total Cost (100-ep moving window) vs Timesteps")
    plt.savefig(f"./models/{WANDB_CONFIG['experiment_name']}/training_curve_total_costs.png")
