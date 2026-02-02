import ast
import os
import sys
from typing import Tuple

import gymnasium
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from newenv import AwesimCityEnv
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode, VectorEnv
from ppo import Actor, ActorBase, Critic, PPO


class ModelArchitecture(nn.Module):

    def __init__(self, env: AwesimCityEnv, dim_out: int, d_model: int, num_decoder_layers: int, num_heads: int):
        super().__init__()
        self.feature_dim = env.feature_dim
        self.entity_count = env.entity_count
        self.car_feature_dim = env.car_feature_dim
        self.car_count = env.car_count
        self.road_feature_dim = env.road_feature_dim
        self.road_count = env.road_count
        self.intersection_feature_dim = env.intersection_feature_dim
        self.intersection_count = env.intersection_count
        self.misc_state_feature_dim = env.misc_state_feature_dim
        self.misc_state_entity_count = env.misc_state_entity_count

        dim_ff = d_model * 4
        self.proj_agent = nn.Sequential(
            nn.Linear(self.car_feature_dim, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_npcs = nn.Sequential(
            nn.Linear(self.car_feature_dim, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_roads = nn.Sequential(
            nn.Linear(self.road_feature_dim, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_intersections = nn.Sequential(
            nn.Linear(self.intersection_feature_dim, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_misc_states = nn.Sequential(
            nn.Linear(self.misc_state_feature_dim, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.agent_embedding = torch.nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.agent_embedding.data, 0, 0.02)
        self.npcs_embedding = torch.nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.npcs_embedding.data, 0, 0.02)
        self.roads_embedding = torch.nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.roads_embedding.data, 0, 0.02)
        self.intersections_embedding = torch.nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.intersections_embedding.data, 0, 0.02)
        self.misc_states_embedding = torch.nn.Parameter(torch.empty(1, 1, d_model), requires_grad=True)
        nn.init.normal_(self.misc_states_embedding.data, 0, 0.02)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, batch_first=True, norm_first=True, dropout=0, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.LayerNorm(dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, dim_out),
        )

    def _get_cars_matrix_from_state(self, state):
        # extract cars from last two dims of state (which may have a batch dim)
        return state[..., :self.car_count, :self.car_feature_dim]
    
    def _get_roads_matrix_from_state(self, state):
        return state[..., self.car_count:self.car_count + self.road_count, :self.road_feature_dim]
    
    def _get_intersections_matrix_from_state(self, state):
        return state[..., self.car_count + self.road_count:self.car_count + self.road_count + self.intersection_count, :self.intersection_feature_dim]
    
    def _get_misc_state_matrix_from_state(self, state):
        return state[..., -self.misc_state_entity_count:, :self.misc_state_feature_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: (batch_size, entity_count, feature_dim)

        cars = self._get_cars_matrix_from_state(x)                   # (batch_size, num_cars, car_feature_dim)
        agent = cars[:, 0:1, :]                                       # (batch_size, 1, car_feature_dim)
        npcs = cars[:, 1:, :]                                        # (batch_size, num_cars-1, car_feature_dim)
        roads = self._get_roads_matrix_from_state(x)                 # (batch_size, num_roads, road_feature_dim)
        intersections = self._get_intersections_matrix_from_state(x) # (batch_size, num_intersections, intersection_feature_dim)
        misc_states = self._get_misc_state_matrix_from_state(x)      # (batch_size, num_misc_states, misc_state_feature_dim)
        
        # projs
        x_agent = self.proj_agent(agent)                             # (batch_size, 1, d_model)
        x_npcs = self.proj_npcs(npcs)                               # (batch_size, num_cars-1, d_model)
        x_roads = self.proj_roads(roads)                             # (batch_size, num_roads, d_model)
        x_intersections = self.proj_intersections(intersections)     # (batch_size, num_intersections, d_model)
        x_misc_states = self.proj_misc_states(misc_states)           # (batch_size, num_misc_states, d_model)

        # add embeddings
        x_agent = x_agent + self.agent_embedding                     # (batch_size, 1, d_model)
        x_npcs = x_npcs + self.npcs_embedding                       # (batch_size, num_cars-1, d_model)
        x_roads = x_roads + self.roads_embedding                     # (batch_size, num_roads, d_model)
        x_intersections = x_intersections + self.intersections_embedding # (batch_size, num_intersections, d_model)
        x_misc_states = x_misc_states + self.misc_states_embedding   # (batch_size, num_misc_states, d_model)

        # cat everything but agent
        x_context = torch.cat([x_npcs, x_roads, x_intersections, x_misc_states], dim=1)  # (batch_size, entity_count-1, d_model)

        # decoder agent in context of others
        x_agent = self.decoder(tgt=x_agent, memory=x_context)       # (batch_size, 1, d_model)

        # take agent token and apply mlp
        x = x_agent.squeeze(1)                                      # (batch_size, d_model)
        x = self.mlp(x)                                             # (batch_size, dim_out)

        return x


WANDB_CONFIG = {
    "no_wandb": False,
    "project": "AwesimCity",
    "experiment_name": "AwesimCity-Dev",
    "notes": """Attention-based model with transformer decoder layers.""",
    "video_interval": 25,
    "video_seconds_max": 20 * 60,       # max seconds of video to record
    "model_save_interval": 25,
}
ENV_CONFIG = {
    "deprecated_map": False,           # use the old small map if True
    "npc_rogue_factor": 0.0,           # fraction of NPC cars that behave more aggressively
    "city_width": 1500,                 # in meters
    "num_cars": 256,                    # number of other cars in the environment
    "decision_interval": 1.0,           # in seconds, how often the agent can take an action (reconfigure the driving assist), or how often the acceleration is applied in non-DAS mode
    "sim_duration": 60 * 60,            # 60 minutes max duration after which the episode ends and wage for entire 8-hour workday is lost
    "min_goal_manhat": 500,              # minimum manhattan distance to goal at start of episode
    "init_fuel_gallons": 3.0,           # initial fuel in gallons -- 1/4th of a 12-gallon tank car
    "success_reward": 1.0,
    "fail_reward": -1.0,
    "crash_causes_fail": True,
    "reward_shaping": True,
    "reward_shaping_gamma": 1.0,
    "reward_shaping_rew_per_meter": 0.01,
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
    "d_model": 128,
    "num_decoder_layers": 3,
    "num_heads": 4,
}
POLICY_CONFIG = {
    "norm_obs": False,   # by running mean and std of observations
}


def make_env(env_index: int = 0) -> AwesimCityEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimCityEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def setup_training_env() -> VectorEnv:
    if VEC_ENV_CONFIG["async"]:
        return AsyncVectorEnv([lambda i=i: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        return SyncVectorEnv([lambda i=i: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)


def train_model(load_actor=None, load_critic=None) -> Tuple[ActorBase, Critic, PPO]:
    """Train the PPO model with the specified configuration."""

    env = make_env()
    vec_env = setup_training_env()

    if not WANDB_CONFIG["no_wandb"]:
        wandb.init(
            project=WANDB_CONFIG["project"],
            name=WANDB_CONFIG["experiment_name"],
            config={**ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG, **NET_CONFIG},
            notes=WANDB_CONFIG["notes"],
            save_code=True,
        )

    obs_space: Box = vec_env.single_observation_space       # type: ignore
    print(f"Observation space: {obs_space}")
    action_space: Discrete = vec_env.single_action_space      # type: ignore
    print(f"Action space: {action_space}")
    print(f"Action meanings: {env.action_meanings}")

    actor_model = ModelArchitecture(env, dim_out=int(action_space.n), **NET_CONFIG)
    total_params = sum(p.numel() for p in actor_model.parameters())
    trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    print("Actor model:")
    print(actor_model)
    print(f"Total parameters: {total_params}, trainable parameters: {trainable_params}")
    critic_model = ModelArchitecture(env, dim_out=1, **NET_CONFIG)

    actor = Actor(actor_model, obs_space, action_space, norm_obs=POLICY_CONFIG["norm_obs"])    # type: ignore
    critic = Critic(critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    if PPO_CONFIG["cmdp_mode"]:
        cost_critic_model = ModelArchitecture(env, dim_out=1, **NET_CONFIG)
        cost_critic = Critic(cost_critic_model, obs_space, norm_obs=POLICY_CONFIG["norm_obs"])
    else:
        cost_critic = None

    if load_actor is not None:
        print(f"Loading actor model from {load_actor}")
        actor.load_state_dict(torch.load(load_actor, map_location=torch.device(PPO_CONFIG["training_device"])))
    if load_critic is not None:
        print(f"Loading critic model from {load_critic}")
        critic.load_state_dict(torch.load(load_critic, map_location=torch.device(PPO_CONFIG["training_device"])))

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
    video_dir = f"videos/{WANDB_CONFIG['experiment_name']}"
    if WANDB_CONFIG["video_interval"] is not None:
        os.makedirs(video_dir, exist_ok=True)
        env = gymnasium.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix="training", video_length=int(WANDB_CONFIG["video_seconds_max"] / ENV_CONFIG["decision_interval"]), fps=int(1 / ENV_CONFIG["decision_interval"]))
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

    load_config = {
        "load_actor": None,
        "load_critic": None,
    }
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
                NET_CONFIG[k] = v   # type: ignore
                found_key = True
            if k in POLICY_CONFIG:
                POLICY_CONFIG[k] = v    # type: ignore
                found_key = True
            if k in WANDB_CONFIG:
                WANDB_CONFIG[k] = v
                found_key = True
            if k in load_config:
                load_config[k] = v      # type: ignore
                found_key = True
            if not found_key:
                # print in yellow color
                print(f"\033[93mWarning: key {k} not found in any config, ignoring the override.\033[0m")

    print("Final configuration:")
    CONFIG = {**WANDB_CONFIG, **ENV_CONFIG, **VEC_ENV_CONFIG, **PPO_CONFIG, **POLICY_CONFIG, **NET_CONFIG}
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    actor, critic, ppo = train_model(**load_config)
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
