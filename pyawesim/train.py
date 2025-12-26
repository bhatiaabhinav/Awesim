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
from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode, VectorEnv
from ppo import Actor, ActorBase, Critic, PPO


def activation_fn(act: str):
    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation function: {act}")


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, groupnorm, num_groups, act):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gn = None if not groupnorm else nn.GroupNorm(min(num_groups, c_out), c_out)
        self.act = activation_fn(act) if act is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBlockDeepDownsample(nn.Module):
    def __init__(self, c_in, c_out, groupnorm, num_groups, act, residual=False):
        super().__init__()
        self.residual = residual
        self.conv1 = ConvBlock(c_in, c_in, kernel_size=3, stride=1, padding=1, groupnorm=groupnorm, num_groups=num_groups, act=act)  # preserve spatial size
        self.conv2 = ConvBlock(c_in, c_out, kernel_size=3, stride=2, padding=1, groupnorm=groupnorm, num_groups=num_groups, act=None)  # downsample by 2 and increase channels
        if self.residual:
            self.shortcut_conv = ConvBlock(c_in, c_out, kernel_size=3, stride=2, padding=1, groupnorm=groupnorm, num_groups=num_groups, act=None)
        self.act = activation_fn(act)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual:
            out += self.shortcut_conv(x)
        out = self.act(out)
        return out


class ModelArchitecture(nn.Module):

    def __init__(self, hidden_dim_per_image, hidden_dim: int, dim_out: int, c_in: int = 3, c_base: int = 32, num_images: int = 9, deep_downsample=False, residual=False, groupnorm=False, num_groups=32, act='relu'):
        super().__init__()
        self.hidden_dim_per_imagestack = hidden_dim_per_image
        self.hidden_dim = hidden_dim
        self.num_images = num_images
        self.num_cams = num_images - 2  # cameras + minimap + speedometer etc.
        self.c_in = c_in    # per cam

        if c_in % 3 != 0:
            raise ValueError(
                "c_in must be divisible by 3 (stacked RGB frames)")
        self.framestack_per_cam = c_in // 3

        c1, c2, c3, c4, c5 = c_base, c_base * 2, c_base * 4, c_base * 4, c_base * 4
        self.flatten_out = c5 * 6 * 6

        def downsample_block(c_in, c_out):
            if deep_downsample:
                return ConvBlockDeepDownsample(c_in, c_out, groupnorm, num_groups, act, residual)
            else:
                return ConvBlock(c_in, c_out, 4, 2, 1, groupnorm, num_groups, act)

        # Shared stem: first two conv layers applied separately to each frame for each camera. All frames are in the batch dimension. No temporal mixing here, only spatial feature extraction.
        self.stem_cams = nn.Sequential(
            ConvBlock(3, c1, 5, 2, 2, groupnorm, num_groups, act),  # 128x128x3 -> 64x64xc1
            downsample_block(c1, c2)   # 64x64xc1 -> 32x32xc2
        )

        # Remaining network operating on temporally stacked features. Different cameras are in the batch dimension now, so spatio-temporal mixing happens for each camera separately, using shared weights.
        self.cnn_cams = nn.Sequential(
            ConvBlock(c2 * self.framestack_per_cam, c2, 1, 1, 0, groupnorm, num_groups, act),  # 1x1 conv to compress temporal info
            downsample_block(c2, c3),  # 32x32x(c2) -> 16x16x(c3)
            downsample_block(c3, c4),  # 16x16x(c3) -> 8x8x(c4)
            ConvBlock(c4, c5, 3, 1, 0, groupnorm, num_groups, act),  # 8x8x(c4) -> 6x6x(c5)
            nn.Flatten(),
        )

        self.hidden_cams = nn.ModuleList([nn.Sequential(nn.Linear(self.flatten_out, self.hidden_dim_per_imagestack), activation_fn(act)) for _ in range(self.num_cams)])

        # for minimap, need only the latest frame in the stack (there is no useful temporal info in minimap, so we discard other frames)
        self.cnn_minimap = nn.Sequential(
            ConvBlock(3, c1, 5, 2, 2, groupnorm, num_groups, act),  # 128x128x3 -> 64x64xc1
            downsample_block(c1, c2),  # 64x64xc1 -> 32x32xc2
            downsample_block(c2, c3),  # 32x32x(c2) -> 16x16x(c3)
            downsample_block(c3, c4),  # 16x16x(c3) -> 8x8x(c4)
            ConvBlock(c4, c5, 3, 1, 0, groupnorm, num_groups, act),  # 8x8x(c4) -> 6x6x(c5)
            nn.Flatten(),
        )
        self.hidden_minimap = nn.Sequential(nn.Linear(self.flatten_out, self.hidden_dim_per_imagestack), activation_fn(act))

        # for info panel, keep only the latest frame in the stack
        self.cnn_info = nn.Sequential(
            ConvBlock(3, c1, 5, 2, 2, groupnorm, num_groups, act),  # 128x128x3 -> 64x64xc1
            downsample_block(c1, c2),  # 64x64xc1 -> 32x32xc2
            downsample_block(c2, c3),  # 32x32x(c2) -> 16x16x(c3)
            downsample_block(c3, c4),  # 16x16x(c3) -> 8x8x(c4)
            ConvBlock(c4, c5, 3, 1, 0, groupnorm, num_groups, act),  # 8x8x(c4) -> 6x6x(c5)
            nn.Flatten(),
        )
        self.hidden_info = nn.Sequential(nn.Linear(self.flatten_out, self.hidden_dim_per_imagestack), activation_fn(act))

        # everything fused
        self.hidden_fused = nn.Linear(num_images * hidden_dim_per_image, hidden_dim)
        self.relu = nn.ReLU()
        self.final_fc = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (self.num_images, self.c_in, 128, 128), f"Expected input shape ({x.shape[0]}, {self.num_images}, {self.c_in}, 128, 128), but got {x.shape}"

        B, N, C, H, W = x.shape

        # do cameras first:
        cam_x = x[:, :self.num_cams, :, :, :]   # (B, num_cams, c_in, 128, 128)

        # stem using shared weights for spatial feature extraction
        cam_x = cam_x.reshape(-1, 3, H, W)         # (B * num_cams * framestack_per_cam, 3, 128, 128)
        cam_x = self.stem_cams(cam_x)           # (B * num_cams * framestack_per_cam, c2, 32, 32)

        # temporal mixing per cam using shared weights
        cam_x = cam_x.view(B * self.num_cams, -1, cam_x.shape[2], cam_x.shape[3])  # (B * num_cams, c2 * framestack_per_cam, 32, 32)
        cam_x = self.cnn_cams(cam_x)                                               # (B * num_cams, flatten_out)

        # per-cam hidden and conca
        cam_x = cam_x.view(B, self.num_cams, -1)               # (B, num_cams, flatten_out)
        cam_x = torch.concat([self.hidden_cams[i](cam_x[:, i, :]) for i in range(self.num_cams)], dim=1)  # (B, num_cams * hidden_dim_per_imagestack)

        # do minimap
        minimap_x = x[:, -2, :, :, :]                     # (B, c_in, 128, 128)
        minimap_x = minimap_x[:, -3:, :, :]               # (B, 3, 128, 128) -- keep only latest frame
        minimap_x = self.cnn_minimap(minimap_x)           # (B, flatten_out)
        minimap_x = self.hidden_minimap(minimap_x)        # (B, hidden_dim_per_imagestack)

        # do info panel
        info_x = x[:, -1, :, :, :]                        # (B, c_in, 128, 128)
        info_x = info_x[:, -3:, :, :]                     # (B, 3, 128, 128) -- keep only latest frame
        info_x = self.cnn_info(info_x)                    # (B, flatten_out)
        info_x = self.hidden_info(info_x)                 # (B, hidden_dim_per_imagestack)

        # fuse all
        x = torch.concat([cam_x, minimap_x, info_x], dim=1)  # (B, num_images * hidden_dim_per_imagestack)
        x = self.hidden_fused(x)                             # (B, hidden_dim)
        x = self.relu(x)

        # final output
        x = self.final_fc(x)                              # (B, dim_out)

        return x


WANDB_CONFIG = {
    "no_wandb": False,
    "project": "Awesim-ApppointmentWorld",
    "experiment_name": "Awesim-Vision",
    "notes": """Shared CNN for cams: Some spatial processing -> temporal mixing -> spatio-temporal processing. Canon hyperparameters. Dense, shaped rewards.""",
    "video_interval": 25,
    "model_save_interval": 25,
}
ENV_CONFIG = {
    "use_das": True,                   # use driving assistance system (DAS) with speed target, time headway target, indicator and stop at intersection controls. Else, use acceleration control.
    "cam_resolution": (128, 128),       # width, height
    "framestack": 3,
    "anti_aliasing": True,
    "goal_lane": 84,
    "city_width": 1000,                 # in meters
    "num_cars": 256,                    # number of other cars in the environment
    "decision_interval": 1.0,             # in seconds, how often the agent can take an action (reconfigure the driving assist), or how often the acceleration is applied in non-DAS mode
    "sim_duration": 60 * 60,            # 60 minutes max duration after which the episode ends and wage for entire 8-hour workday is lost
    "appointment_time": 60 * 20,        # 20 minutes to reach the appointment
    "cost_gas_per_gallon": 4.0,         # in NYC (USD)
    "cost_car": 0,                      # cost of the car (USD), incurred if crashed (irrespective of severity)
    "cost_lost_wage_per_hour": 40,      # of a decent job in NYC (USD)
    "may_lose_entire_day_wage": False,  # lose entire 8-hour workday wage if not reached by the end of sim_duration
    "init_fuel_gallons": 3.0,           # initial fuel in gallons -- 1/4th of a 12-gallon tank car
    "goal_reward": 100.0,                 # A positive number to incentivize reaching the goal
    "crash_penalty": 100.0,               # Dense signal for crashing
    "reward_shaping": True,
    "reward_shaping_gamma": 1.0,
}
VEC_ENV_CONFIG = {
    "n_envs": 16,
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
    "batch_size": 128,
    "nepochs": 4,
    "target_kl": 0.01,
    "lam": 0.3,
    "gamma": 0.999,
    "clipnorm": 0.5,
    "clip_ratio": 0.2,
    "adam_weight_decay": 0.00,
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
NET_CONFIG = {
    "hidden_dim_per_image": 128,
    "hidden_dim": 512,
    "c_base": 32,
    "deep_downsample": False,
    "residual": False,
    "groupnorm": False,
    "num_groups": 32,
    "act": "relu",
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

    NET_CONFIG["c_in"] = ENV_CONFIG["framestack"] * 3

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
    total_params = sum(p.numel() for p in actor_model.parameters())
    trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    print("Actor model:")
    print(actor_model)
    print(f"Total parameters: {total_params}, trainable parameters: {trainable_params}")
    critic_model = ModelArchitecture(dim_out=1, **NET_CONFIG)

    actor = Actor(actor_model, obs_space, action_space, norm_obs=POLICY_CONFIG["norm_obs"], state_dependent_std=POLICY_CONFIG["state_dependent_std"])    # type: ignore
    critic = Critic(critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    if PPO_CONFIG["cmdp_mode"]:
        cost_critic_model = ModelArchitecture(dim_out=1, **NET_CONFIG)
        cost_critic = Critic(cost_critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    else:
        cost_critic = None

    ppo = PPO(vec_env, actor, critic, cost_critic=cost_critic, **PPO_CONFIG, custom_info_keys_to_log_at_episode_end=['crashed', 'reached_goal', 'reached_in_time', 'reached_but_late', 'out_of_fuel', 'timeout', 'total_fuel_consumed', 'crashed_forward', 'crashed_backward', 'progress_m_during_crash', 'progress_m_during_front_crash', 'progress_m_during_back_crash', 'crashed_on_intersection', 'aeb_in_progress_during_crash', 'opt_dist_to_goal_m'])  # type: ignore

    # make dir for saving models
    model_dir = f"./models/{WANDB_CONFIG['experiment_name']}"
    os.makedirs(model_dir, exist_ok=True)
    # print the architecture to a file in this directory
    with open(f"{model_dir}/model_architecture.txt", "w") as f:
        f.write("Actor model:\n")
        f.write(str(actor_model))
    if not WANDB_CONFIG["no_wandb"]:
        wandb.save(f"{model_dir}/model_architecture.txt")

    # an env for video recording
    env = None
    video_dir = f"videos/{WANDB_CONFIG['experiment_name']}"
    if WANDB_CONFIG["video_interval"] is not None:
        env = make_env()
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="training", video_length=int(ENV_CONFIG["appointment_time"] / ENV_CONFIG["decision_interval"]), fps=int(1 / ENV_CONFIG["decision_interval"]))
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
                "rollout/opt_dist_to_goal_m_mean": ppo.stats['info_key_means']['opt_dist_to_goal_m'][-1],
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
        if ppo.stats['iterations'] % WANDB_CONFIG["model_save_interval"] == 0:
            torch.save(actor.state_dict(), f"{model_dir}/actor_iter{ppo.stats['iterations']}.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_iter{ppo.stats['iterations']}.pth")
            torch.save(actor.state_dict(), f"{model_dir}/actor_latest.pth")
            torch.save(critic.state_dict(), f"{model_dir}/critic_latest.pth")
            if not WANDB_CONFIG["no_wandb"]:
                wandb.save(f"{model_dir}/actor_latest.pth")

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
