import sys
import torch
from gymenv import AwesimEnv
from gymenv_utils import make_mlp_with_layernorm
from ppo import Actor


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
POLICY_CONFIG = {
    "net_arch": [1024, 512, 256],
}


def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index)  # type: ignore


def test_model(model_path) -> None:

    env = make_env(0)
    env.should_render = True
    env.synchronized = True
    env.synchronized_sim_speedup = 16.0
    env.render_server_ip = "127.0.0.1"
    env.verbose = True

    obs_dim = env.observation_space.shape[0]   # type: ignore
    action_dim = env.action_space.shape[0]     # type: ignore

    actor_model = make_mlp_with_layernorm(obs_dim, POLICY_CONFIG["net_arch"], action_dim)
    actor = Actor(actor_model, env.action_space)    # type: ignore
    actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    actor.evaluate_policy(env, 100)


if __name__ == "__main__":
    test_model(sys.argv[1])
