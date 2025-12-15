import ast
import os
import sys
import gymnasium
import numpy as np
import torch
from gymenv import AwesimEnv
from ppo import Actor
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode
from train import ModelArchitecture

TEST_CONFIG = {
    "speedup": 8.0,                     # simulation speedup factor
    "render": True,                     # whether to render the environment
    "synchronized": True,               # whether to run the simulation in synchronized mode (match sim with real wall time)
    "render_server_ip": "127.0.0.1",    # IP address of the render server
    "verbose": True,                    # whether to print detailed logs
    "parallel": False,                  # whether to run multiple environments in parallel for evaluation
    "episodes": 100,                    # number of episodes to run for evaluation
    "video_interval": 10,               # interval for video recording (in episodes), None to disable
}
ENV_CONFIG = {
    "cam_resolution": (128, 128),       # width, height
    "framestack": 4,
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
    "reward_shaping_gamma": 0.999,
    "deterministic_actions": True,
}
VEC_ENV_CONFIG = {
    "n_envs": 8,
    "async": True,
}
NET_CONFIG = {
    "hidden_dim_per_image": 128,
    "hidden_dim": 512,
    "c_base": 32,
    "cnn_type": "simple",
}
POLICY_CONFIG = {
    "norm_obs": True,
    "state_dependent_std": True,
    "deterministic": True,
}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def test_model(model_path) -> None:
    NET_CONFIG["c_in"] = ENV_CONFIG["framestack"] * 3

    env = make_env(0)
    env.should_render = TEST_CONFIG["render"]
    env.synchronized = TEST_CONFIG["synchronized"]
    env.synchronized_sim_speedup = TEST_CONFIG["speedup"]
    env.render_server_ip = TEST_CONFIG["render_server_ip"]
    env.verbose = TEST_CONFIG["verbose"]

    obs_dim = env.observation_space.shape[0]   # type: ignore
    action_dim = env.action_space.shape[0]     # type: ignore

    actor_dim_out = action_dim * 2 if POLICY_CONFIG["state_dependent_std"] else action_dim
    actor_model = ModelArchitecture(dim_out=actor_dim_out, **NET_CONFIG)
    
    actor = Actor(actor_model, env.observation_space, env.action_space, deterministic=POLICY_CONFIG["deterministic"], norm_obs=POLICY_CONFIG["norm_obs"], state_dependent_std=POLICY_CONFIG["state_dependent_std"])    # type: ignore
    
    # Load the model
    print(f"Loading model from {model_path}")
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if TEST_CONFIG["parallel"]:
        vec_cls = AsyncVectorEnv if VEC_ENV_CONFIG["async"] else SyncVectorEnv
        vec_env = vec_cls([(lambda i=i: make_env(i)) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)
        rews, costs, lens, succs = actor.evaluate_policy_parallel(vec_env, TEST_CONFIG["episodes"], deterministic=POLICY_CONFIG["deterministic"])
        vec_env.close()
    else:
        # set up env for video recording as well periodically
        if TEST_CONFIG["video_interval"] is not None:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            env = gymnasium.wrappers.RecordVideo(env, video_folder=f"videos/{model_name}", episode_trigger=lambda x: x % TEST_CONFIG["video_interval"] == 0, name_prefix="test")
        rews, costs, lens, succs = actor.evaluate_policy(env, TEST_CONFIG["episodes"], deterministic=POLICY_CONFIG["deterministic"])

    print(f"Average Reward: {np.mean(rews)}, Average Cost: {np.mean(costs)}, Average Length: {np.mean(lens)}, Success Rate: {np.mean(succs)}")

    env.close()


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
            if k in TEST_CONFIG:
                TEST_CONFIG[k] = v
                found_key = True
            elif k in ENV_CONFIG:
                ENV_CONFIG[k] = v
                found_key = True
            elif k in NET_CONFIG:
                NET_CONFIG[k] = v
                found_key = True
            elif k in POLICY_CONFIG:
                POLICY_CONFIG[k] = v    # type: ignore
                found_key = True
            
            if not found_key:
                # print in yellow color
                print(f"\033[93mWarning: key {k} not found in any config, ignoring the override.\033[0m")

    if len(sys.argv) < 2 or "=" in sys.argv[1]:
        print("Usage: python test.py <model_path> [key=value ...]")
        sys.exit(1)

    model_path = sys.argv[1]
    test_model(model_path)
