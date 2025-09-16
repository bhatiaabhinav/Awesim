import os
import sys
from typing import Union

from gymenv import AwesimEnv
from gymenv_utils import POLICY_NET_1024_512_256_RELU_LN_CONFIG, POLICY_NET_1024_512_256_RELU_LN_SQUASH_OUTPUT_CONFIG
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from squashed_policy import SquashedActorCriticPolicy

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
POLICY_CLS = SquashedActorCriticPolicy
POLICY_CONFIG = POLICY_NET_1024_512_256_RELU_LN_SQUASH_OUTPUT_CONFIG

def make_env(env_index: int = 0) -> AwesimEnv:
    """Create an AwesimEnv instance with specified configuration."""
    return AwesimEnv(**ENV_CONFIG, env_index=env_index, verbose=True)  # type: ignore

def setup_evaluation_env(load_model_experiment_name, load_model_step: int) -> tuple[Union[VecNormalize, VecEnv], str]:
    """Set up environment for evaluation with loaded model and normalization stats."""
    model_path = f"models/{load_model_experiment_name}/{load_model_experiment_name}_{load_model_step}_steps.zip"
    vec_stats_path = f"models/{load_model_experiment_name}/{load_model_experiment_name}_vecnormalize_{load_model_step}_steps.pkl"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    env = make_env()
    env.should_render = True
    env.synchronized = True
    env.synchronized_sim_speedup = 16.0
    env.render_server_ip = "127.0.0.1"
    vec_env = make_vec_env(lambda: env)
    
    if os.path.exists(vec_stats_path):
        print(f"Loading VecNormalize stats from {vec_stats_path}")
        vec_env = VecNormalize.load(vec_stats_path, vec_env)
        vec_env.training = False
        print("VecNormalize stats loaded")
    else:
        print(f"Warning: VecNormalize stats not found at {vec_stats_path}. Proceeding without normalization.")
    
    return vec_env, model_path



def evaluate_model(load_model_experiment_name, load_model_step: int) -> None:
    """Evaluate the model with the specified checkpoint."""
    vec_env, model_path = setup_evaluation_env(load_model_experiment_name, load_model_step)
    model = PPO(POLICY_CLS, vec_env, policy_kwargs=POLICY_CONFIG)

    print(f"Loading model from {model_path}")
    model.set_parameters(model_path)
    print("Model loaded")
    
    obs = vec_env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs[0] if isinstance(obs, tuple) else obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)


if __name__ == "__main__":
    load_model_experiment_name = sys.argv[1]
    load_model_step = int(sys.argv[2])
    evaluate_model(load_model_experiment_name, load_model_step)