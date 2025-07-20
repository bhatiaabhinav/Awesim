from typing import Any, SupportsFloat, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bindings import *

class AwesimEnv(gym.Env):
    def __init__(self, city_width: int = 1000, num_cars: int = 256, decision_interval: float = 0.1, sim_duration: float = 60 * 60) -> None:
        super().__init__()
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.sim_duration = sim_duration

        self.action_space = spaces.Discrete(6) # speed inc 5, speed dec 5, merge left, merge right, cruise, cruise_adaptive
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)  # position, speed (div 100 mph), lane_left_exists, lane_right_exists, next_car_exists, next_car_distance (normalized by current lane length), next_car_speed_div_100mph
        self.sim = sim_malloc()
        # sim_init(self.sim)
        # sim_connect_to_render_server(self.sim, "127.0.0.1", 4242)
        sim_set_agent_enabled(self.sim, True)
        self.agent: Car = sim_get_agent_car(self.sim)

    def _get_observation(self):
        situation = sim_get_situational_awareness(self.sim, 0)
        progress = car_get_lane_progress(self.agent)
        speed_div_100 = to_mph(car_get_speed(self.agent)) / 100
        lane_left_exists = situation.lane_left is not None
        lane_right_exists = situation.lane_right is not None
        next_car_exists = situation.is_vehicle_ahead
        lane_length = lane_get_length(situation.lane)
        next_car_distance = situation.distance_to_lead_vehicle / lane_length if next_car_exists else 0.0
        next_car_speed_div_100 = to_mph(car_get_speed(situation.lead_vehicle)) / 100 if next_car_exists else 0.0
        return np.asarray([progress, speed_div_100, lane_left_exists, lane_right_exists, next_car_exists, next_car_distance, next_car_speed_div_100], dtype=np.float32)
    

    def _get_reward(self):
        # if there is no lead vehicle, we want to encourage the agent to drive faster. If there is, we want the agent to maintain 3 seconds distance to the lead vehicle
        situation = sim_get_situational_awareness(self.sim, 0)
        my_speed = car_get_speed(self.agent)
        if not situation.is_vehicle_ahead:
            return to_mph(my_speed) / 100.0  # Encourage driving faster when no lead vehicle is present
        else:
            # Maintain 3 seconds distance to the lead vehicle
            desired_distance = my_speed * 3.0  # 3 seconds distance
            actual_distance = situation.distance_to_lead_vehicle
            return -abs(desired_distance - actual_distance)  # Penalize for not maintaining distance
        
    def _should_terminate(self):
        # detect crash with lead vehicle``
        situation = sim_get_situational_awareness(self.sim, 0)
        if situation.is_vehicle_ahead:
            distance_center_to_center = situation.distance_to_lead_vehicle
            distance_bumper_to_tail = distance_center_to_center - (car_get_length(self.agent) / 2 + car_get_length(situation.lead_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                print(f"Agent {self.agent.id} crashed into lead vehicle #{situation.lead_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        return False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, clock_reading(0, 8, 0, 0), WEATHER_SUNNY)
        sim_set_synchronized(self.sim, True, 1.0)
        sim_connect_to_render_server(self.sim, "127.0.0.1", 4242)

        sim_set_agent_enabled(self.sim, True)
        self.agent = sim_get_agent_car(self.sim)
        
        situational_awareness_build(self.sim, self.agent.id)
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        t0 = sim_get_time(self.sim)
        situational_awareness_build(self.sim, self.agent.id)
        ongoing_procedure = sim_get_ongoing_procedure(self.sim, self.agent.id)
        status = PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED
        if action == 0 or action == 1:
            speed_delta_mag = from_mph(5)
            speed_delta_direction = 1 if action == 0 else -1
            speed_delta = speed_delta_direction * speed_delta_mag
            print(f"Initializing ADJUST_SPEED procedure with speed delta {speed_delta} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_ADJUST_SPEED, [car_get_speed(self.agent) + speed_delta, from_mph(0.5), 5.0, 1.0])  # Arguments: desired speed, tolerance, timeout duration, whether to use preferred acceleration profile (0 or 1)
        elif action == 2 or action == 3:
            merge_dir = INDICATOR_LEFT if action == 2 else INDICATOR_RIGHT
            speed = car_get_speed(self.agent)
            print(f"Initializing MERGE procedure with direction {merge_dir}...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_MERGE, [float(merge_dir), 5.0, speed, 0.0, 2.0, 1.0])  # Arguments: CarIndicator direction (left/right), timeout_duration, desired cruise speed, whether cruise is adaptive, follow distance in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)
        elif action == 4 or action == 5:
            speed = car_get_speed(self.agent)
            adaptive = 1.0 if action == 5 else 0.0
            print(f"Initializing CRUISE procedure with adaptive={adaptive} and speed={speed} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_CRUISE, [5.0, speed, adaptive, 3.0, 1.0])  # Arguments: duration, desired cruise speed, whether adaptive, follow duration in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)

        if status == PROCEDURE_STATUS_INITIALIZED:
            while (status in [PROCEDURE_STATUS_INITIALIZED, PROCEDURE_STATUS_IN_PROGRESS]):
                status = procedure_step(self.sim, self.agent, ongoing_procedure)
                if status not in [PROCEDURE_STATUS_IN_PROGRESS, PROCEDURE_STATUS_COMPLETED]:
                    print(f"Agent {self.agent.id} failed to execute procedure. Status: {status}")
                    break   # Note: Even though procedure_step will recommend some controls even when the procedure already succeeded, we do not apply it since we want the agent to be able to apply new procedures right away.
                simulate(self.sim, self.decision_interval)
                situational_awareness_build(self.sim, self.agent.id)
        else:
            print(f"Agent {self.agent.id} failed to initialize procedure. Status: {status}")
            car_reset_all_control_variables(self.agent)
            simulate(self.sim, self.decision_interval)  # simulate the step even if procedure initialization failed
            situational_awareness_build(self.sim, self.agent.id)

        t1 = sim_get_time(self.sim)

        obs = self._get_observation()
        reward = self._get_reward() * (t1 - t0)  # Normalize reward by the time taken for the step, to avoid bias towards shorter steps
        terminated = self._should_terminate()
        truncated = sim_get_time(self.sim) >= self.sim_duration
        info = {"status": status, "time": sim_get_time(self.sim)}
        return obs, reward, terminated, truncated, info
    
    def close(self) -> None:
        sim_disconnect_from_render_server(self.sim)
        sim_free(self.sim)
        super().close()

        


# Example usage
if __name__ == "__main__":
    # Create environment
    env = AwesimEnv(
        city_width=1000,
        num_cars=256,
        decision_interval=0.1,
        sim_duration=60.0
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
            # break

            print(f"Step {step_count}: reward={reward:.3f}, total_reward={total_reward:.3f}")
            # print(f"  Distance to lead vehicle: {info['distance_to_lead_vehicle']:.2f}m")

            if terminated or truncated:
                print(f"Episode finished after {step_count} steps")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        env.close()
        print("Environment closed")
