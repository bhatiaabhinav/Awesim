import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from cffi import FFI
from typing import Dict, Any, Tuple, Optional


class AwesimEnv(gym.Env):
    """
    Gymnasium environment for the Awesim driving simulator.

    Action Space:
        - acceleration: continuous value (typically -10.0 to 10.0 m/s²)
        - lane_indicator: discrete (0=NONE, 1=LEFT, 2=RIGHT)
        - turn_indicator: discrete (0=NONE, 1=LEFT, 2=RIGHT)

    Observation Space:
        Dictionary containing various situational awareness variables
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self,
                 city_width: int = 1000,
                 num_cars: int = 256,
                 decision_interval: float = 0.1,
                 max_episode_time: float = 100.0,
                 render_mode: Optional[str] = None,
                 render_server_ip: str = "127.0.0.1",
                 render_server_port: int = 4242):

        super().__init__()

        # Environment parameters
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.max_episode_time = max_episode_time
        self.render_mode = render_mode
        self.render_server_ip = render_server_ip
        self.render_server_port = render_server_port

        # Initialize CFFI
        self._init_cffi()

        # Action space: [acceleration, lane_indicator, turn_indicator]
        self.action_space = spaces.Dict({
            'acceleration': spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
            'lane_indicator': spaces.Discrete(3),  # 0=NONE, 1=LEFT, 2=RIGHT
            'turn_indicator': spaces.Discrete(3)   # 0=NONE, 1=LEFT, 2=RIGHT
        })

        # Observation space (will be updated after first reset)
        # This is a placeholder - actual observation space depends on situational_awareness_build output
        self.observation_space = spaces.Dict({
            'distance_to_lead_vehicle': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=0.0, high=200.0, shape=(1,), dtype=np.float32),
            'position_x': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'position_y': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'simulation_time': spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Simulation state
        self.sim = None
        self.agent = None
        self.current_time = 0.0
        self.episode_reward = 0.0

    def _init_cffi(self):
        """Initialize CFFI and load the simulation library."""
        self.ffi = FFI()

        # Load header file
        try:
            with open(os.path.join("python", "libawesim.h")) as f:
                self.ffi.cdef(f.read(), override=True)
        except FileNotFoundError:
            raise FileNotFoundError("libawesim.h not found in python/ directory")

        # Load dynamic library
        lib_extension = "dll" if os.name == "nt" else "so"
        lib_name = f"libawesim.{lib_extension}"
        lib_path = os.path.abspath(os.path.join("bin", lib_name))

        try:
            self.libsim = self.ffi.dlopen(lib_path)
        except OSError as e:
            raise OSError(f"Error loading library {lib_path}: {e}")

    def _get_indicator_constant(self, indicator_type: str, value: int) -> int:
        """Convert discrete action to simulator constant."""
        if indicator_type == "lane":
            if value == 0:
                return self.libsim.INDICATOR_NONE
            elif value == 1:
                return self.libsim.INDICATOR_LEFT
            elif value == 2:
                return self.libsim.INDICATOR_RIGHT
        elif indicator_type == "turn":
            if value == 0:
                return self.libsim.INDICATOR_NONE
            elif value == 1:
                return self.libsim.INDICATOR_LEFT
            elif value == 2:
                return self.libsim.INDICATOR_RIGHT

        return self.libsim.INDICATOR_NONE

    def _extract_observation(self, situation) -> Dict[str, np.ndarray]:
        """Extract observation from situational awareness structure."""
        # This is a basic extraction - you may need to modify based on
        # the actual fields available in your situational_awareness structure
        obs = {
            'distance_to_lead_vehicle': np.array([situation.distance_to_lead_vehicle], dtype=np.float32),
            'simulation_time': np.array([self.libsim.sim_get_time(self.sim)], dtype=np.float32),
        }

        # Add more fields as available in your situation structure
        # For example:
        # obs['speed'] = np.array([situation.speed], dtype=np.float32)
        # obs['position_x'] = np.array([situation.position_x], dtype=np.float32)
        # obs['position_y'] = np.array([situation.position_y], dtype=np.float32)

        return obs

    def _calculate_reward(self, situation, action: Dict[str, Any]) -> float:
        """Calculate reward based on current situation and action taken."""
        reward = 0.0

        # Example reward function - customize based on your objectives

        # Penalty for being too close to lead vehicle
        if situation.distance_to_lead_vehicle < 10.0:  # Less than 10 meters
            reward -= 10.0
        elif situation.distance_to_lead_vehicle < 30.0:  # Less than 30 meters
            reward -= 1.0

        # Small positive reward for maintaining safe distance
        if 30.0 <= situation.distance_to_lead_vehicle <= 100.0:
            reward += 0.1

        # Penalty for excessive acceleration/deceleration
        abs_acceleration = abs(action['acceleration'][0])
        if abs_acceleration > 5.0:
            reward -= abs_acceleration * 0.1

        # Small positive reward for progressing through time (staying alive)
        reward += 0.01

        return reward

    def _is_terminated(self, situation) -> bool:
        """Check if episode should terminate."""
        # Terminate if collision or other terminal condition
        # You'll need to check what fields are available in situation

        # Example termination conditions:
        # if hasattr(situation, 'collision') and situation.collision:
        #     return True

        # Terminate if too close to lead vehicle (potential collision)
        if situation.distance_to_lead_vehicle < 2.0:
            return True

        return False

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self.current_time >= self.max_episode_time

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Clean up previous simulation if exists
        if self.sim is not None:
            if self.render_mode == "human":
                self.libsim.sim_disconnect_from_render_server(self.sim)
            self.libsim.sim_free(self.sim)

        # Create new simulation
        self.sim = self.libsim.sim_malloc()

        # Setup simulation parameters
        init_clock = self.libsim.clock_reading(0, 8, 0, 0)  # Monday 8:00 AM
        init_weather = self.libsim.WEATHER_SUNNY

        self.libsim.awesim_setup(self.sim, self.city_width, self.num_cars, 0.02,
                                init_clock, init_weather)

        # Enable agent
        self.libsim.sim_set_agent_enabled(self.sim, True)
        self.agent = self.libsim.sim_get_agent_car(self.sim)

        # Setup synchronization
        self.libsim.sim_set_synchronized(self.sim, True, 1.0)

        # Connect to render server if in human mode
        if self.render_mode == "human":
            self.libsim.sim_connect_to_render_server(
                self.sim,
                self.render_server_ip.encode('utf-8'),
                self.render_server_port
            )

        # Reset episode state
        self.current_time = 0.0
        self.episode_reward = 0.0

        # Get initial observation
        situation = self.libsim.situational_awareness_build(self.agent, self.sim)
        observation = self._extract_observation(situation)

        info = {
            'episode_time': self.current_time,
            'episode_reward': self.episode_reward
        }

        return observation, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        if self.sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Apply actions to agent
        acceleration = float(action['acceleration'][0])
        lane_indicator = self._get_indicator_constant("lane", action['lane_indicator'])
        turn_indicator = self._get_indicator_constant("turn", action['turn_indicator'])

        self.libsim.car_set_acceleration(self.agent, acceleration)
        self.libsim.car_set_indicator_lane(self.agent, lane_indicator)
        self.libsim.car_set_indicator_turn(self.agent, turn_indicator)

        # Simulate one step
        self.libsim.simulate(self.sim, self.decision_interval)
        self.current_time = self.libsim.sim_get_time(self.sim)

        # Get new observation
        situation = self.libsim.situational_awareness_build(self.agent, self.sim)
        observation = self._extract_observation(situation)

        # Calculate reward
        reward = self._calculate_reward(situation, action)
        self.episode_reward += reward

        # Check termination conditions
        terminated = self._is_terminated(situation)
        truncated = self._is_truncated()

        info = {
            'episode_time': self.current_time,
            'episode_reward': self.episode_reward,
            'distance_to_lead_vehicle': situation.distance_to_lead_vehicle
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        # Rendering is handled by the simulation's render server
        # when render_mode="human" and connected in reset()
        pass

    def close(self):
        """Clean up environment resources."""
        if self.sim is not None:
            if self.render_mode == "human":
                self.libsim.sim_disconnect_from_render_server(self.sim)
            self.libsim.sim_free(self.sim)
            self.sim = None


# Example usage
if __name__ == "__main__":
    # Create environment
    env = AwesimEnv(
        city_width=1000,
        num_cars=256,
        decision_interval=0.1,
        max_episode_time=30.0,  # Shorter for testing
        render_mode="human"  # Enable visualization
    )

    print("Created Awesim Gymnasium environment")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run a simple episode
    observation, info = env.reset()
    print(f"Initial observation: {observation}")

    total_reward = 0
    step_count = 0

    try:
        while True:
            # Random action for demonstration
            action = env.action_space.sample()

            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            print(f"Step {step_count}: reward={reward:.3f}, total_reward={total_reward:.3f}")
            print(f"  Distance to lead vehicle: {info['distance_to_lead_vehicle']:.2f}m")

            if terminated or truncated:
                print(f"Episode finished after {step_count} steps")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        env.close()
        print("Environment closed")
