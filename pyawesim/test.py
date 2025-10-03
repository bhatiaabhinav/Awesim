import ast
import sys
import numpy as np
import torch
from gymenv import AwesimEnv
from gymenv_utils import make_mlp
from ppo import Actor
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode

TEST_CONFIG = {
    "speedup": 8.0,                     # simulation speedup factor
    "render": True,                     # whether to render the environment
    "synchronized": True,               # whether to run the simulation in synchronized mode (match sim with real wall time)
    "render_server_ip": "127.0.0.1",    # IP address of the render server
    "verbose": True,                    # whether to print detailed logs
    "parallel": False,                  # whether to run multiple environments in parallel for evaluation
    "episodes": 100,                    # number of episodes to run for evaluation
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
    "may_lose_entire_day_wage": True,   # lose entire 8-hour workday wage if not reached by the end of sim_duration
    "init_fuel_gallons": 3.0,           # initial fuel in gallons -- 1/4th of a 12-gallon tank car
    "goal_reward": 1.0,                 # A small positive number to incentivize reaching the goal
    "deterministic_actions": True,      # If True, actions to select follow-mode vs stop-mode and turn/merge signals are deterministic instead of bernoulli-sampled
}
VEC_ENV_CONFIG = {
    "n_envs": 8,
    "async": True,
}
POLICY_CONFIG = {
    "norm_obs": True,
    "net_arch": [1024, 512, 256],
    "layernorm": True,
    "deterministic": True,
}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def test_model(model_path) -> None:

    env = make_env(0)
    env.should_render = TEST_CONFIG["render"]
    env.synchronized = TEST_CONFIG["synchronized"]
    env.synchronized_sim_speedup = TEST_CONFIG["speedup"]
    env.render_server_ip = TEST_CONFIG["render_server_ip"]
    env.verbose = TEST_CONFIG["verbose"]

    obs_dim = env.observation_space.shape[0]   # type: ignore
    action_dim = env.action_space.shape[0]     # type: ignore

    actor_model = make_mlp(obs_dim, POLICY_CONFIG["net_arch"], action_dim, layernorm=POLICY_CONFIG["layernorm"])
    actor = Actor(actor_model, env.action_space, deterministic=POLICY_CONFIG["deterministic"], norm_obs=POLICY_CONFIG["norm_obs"])    # type: ignore
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if TEST_CONFIG["parallel"]:
        vec_cls = AsyncVectorEnv if VEC_ENV_CONFIG["async"] else SyncVectorEnv
        vec_env = vec_cls([lambda: make_env(i + 1) for i in range(VEC_ENV_CONFIG["n_envs"])], copy=False, autoreset_mode=AutoresetMode.SAME_STEP)
        rews, lens, succs = actor.evaluate_policy_parallel(vec_env, TEST_CONFIG["episodes"])
        vec_env.close()
    else:
        rews, lens, succs = actor.evaluate_policy(env, TEST_CONFIG["episodes"])

    print(f"Average Reward: {np.mean(rews)}, Average Length: {np.mean(lens)}, Success Rate: {np.mean(succs)}")

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
            if k in ENV_CONFIG:
                ENV_CONFIG[k] = v
                found_key = True
            if k in POLICY_CONFIG:
                POLICY_CONFIG[k] = v
                found_key = True
            if not found_key:
                # print in yellow color
                print(f"\033[93mWarning: key {k} not found in any config, ignoring the override.\033[0m")

    test_model(sys.argv[1])
