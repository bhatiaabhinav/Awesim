from typing import Any, SupportsFloat, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bindings import *

class AwesimEnv(gym.Env):
    def __init__(self, city_width: int = 1000, num_cars: int = 256, decision_interval: float = 0.1, sim_duration: float = 60 * 60, synchronized: bool = False, synchronized_sim_speedup = 1.0, should_render: bool = False) -> None:
        super().__init__()
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.sim_duration = sim_duration
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render

        self.sim = sim_malloc()
        sim_init(self.sim)
        sim_set_agent_enabled(self.sim, True)
        self.agent: Car = sim_get_new_car(self.sim)
        print(f"Agent car ID: {self.agent.id}")

        # self.action_space = spaces.Discrete(5) # speed inc 10, speed dec 5, merge left, merge right, cruise, cruise_adaptive  # TODO: removing adaptive for a while

        # Action factors:
        # Index 0: Whether to execute NO_OP procedure (> 0 means yes)
        # Index 1: Whether to execute MERGE procedure (> 0 means yes)
        # Index 2: Whether to execute ADJUST_SPEED procedure (> 0 means yes)
        # Index 3: Whether to execute STOP procedure (> 0 means yes)
        # Index 4: Whether to execute CRUISE procedure (> 0 means yes). Ultimately, the selected procedure will be the one with the highest value.
        # Index 5: Speed delta. -1 means -60mph, 0 means no change, 1 means +60mph. Final desired speed will be clamped between [-10mph, top_speed_mph].
        # Index 6: Procedure duration: -1: 0.1, 0: 5 second, 1: 10 seconds.
        # Index 7: Merge direction: <0 means left, >0 means right
        # Index 8: Adaptive cruise: >0 means yes.
        # Index 9: Follow distance in seconds (if adaptive cruise is enabled): -1: 0.1, 0: 3, 1: 6.0
        # Index 10: Whether to use preferred acceleration profile, >0 means yes.
        # Index 11: Whether to leave indicators on during NO_OP procedure, >0 means yes.
        # Index 12: Whether to use linear braking for STOP procedure, >0 means yes, else smooth braking is used.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )

        n_state_factors = len(self._get_observation())
        self.observation_space = spaces.Box(low=-10, high=10, shape=(n_state_factors,), dtype=np.float32)

    def _get_observation(self):
        situation = sim_get_situational_awareness(self.sim, 0)
        progress = car_get_lane_progress(self.agent)
        speed = car_get_speed(self.agent)
        speed_div_100mph = speed / from_mph(60)
        lane_left_exists = situation.lane_left is not None
        lane_right_exists = situation.lane_right is not None
        next_car_exists = situation.is_vehicle_ahead
        next_car_distance_div_100m = situation.distance_to_lead_vehicle / 100 if next_car_exists else 0.0
        speed_plus_eps = speed + 1e-6  if speed == 0 else speed
        relative_speed = speed - car_get_speed(situation.lead_vehicle) if next_car_exists else 0.0
        relative_speed_plus_eps = relative_speed + 1e-6 if relative_speed == 0 else relative_speed
        next_car_distance_ttc_abs_div_10 = np.clip(situation.distance_to_lead_vehicle / (10 * speed_plus_eps), -10, 10) if next_car_exists else 0.0
        next_car_distance_ttc_rel_div_10 = np.clip(situation.distance_to_lead_vehicle / (10 * relative_speed_plus_eps), -10, 10) if next_car_exists else 0.0
        next_car_speed_div_100mph = car_get_speed(situation.lead_vehicle) / from_mph(60) if next_car_exists else 0.0


        return np.asarray([speed_div_100mph, lane_left_exists, lane_right_exists, next_car_exists, next_car_distance_div_100m, next_car_distance_ttc_abs_div_10, next_car_distance_ttc_rel_div_10, next_car_speed_div_100mph], dtype=np.float32)
    

    def _get_reward(self):
        # if there is no lead vehicle, we want to encourage the agent to drive faster. If there is, we want the agent to maintain 3 seconds distance to the lead vehicle
        situation = sim_get_situational_awareness(self.sim, 0)
        my_speed = car_get_speed(self.agent)

        return to_mph(my_speed) / 100.0  # Encourage driving faster

        # if not situation.is_vehicle_ahead:
        #     return to_mph(my_speed) / 100.0  # Encourage driving faster when no lead vehicle is present
        # else:
        #     # Maintain 3 seconds distance to the lead vehicle
        #     desired_distance = my_speed * 3.0  # 3 seconds distance
        #     actual_distance = situation.distance_to_lead_vehicle
        #     difference = actual_distance - desired_distance
        #     difference_in_time = difference / my_speed if my_speed > 0 else 0.0
        #     return -abs(difference_in_time)  # Penalize for not maintaining distance
        
    def _should_terminate(self):
        # detect crash with lead vehicle
        situation = sim_get_situational_awareness(self.sim, 0)
        if situation.is_vehicle_ahead:
            distance_center_to_center = situation.distance_to_lead_vehicle
            distance_bumper_to_tail = distance_center_to_center - (car_get_length(self.agent) / 2 + car_get_length(situation.lead_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                print(f"Agent {self.agent.id} crashed into lead vehicle #{situation.lead_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        return False

    def _should_truncate(self):
        # truncate if simulation time exceeds sim_duration
        return sim_get_time(self.sim) >= self.sim_duration

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # if self.sim is not None:
        #     sim_free(self.sim)
        # self.sim = sim_malloc()
        sim_disconnect_from_render_server(self.sim)
        awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, clock_reading(0, 8, 0, 0), WEATHER_SUNNY)
        if self.synchronized:
            sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            sim_connect_to_render_server(self.sim, "127.0.0.1", 4242)

        sim_set_agent_enabled(self.sim, True)
        self.agent = sim_get_agent_car(self.sim)
        
        situational_awareness_build(self.sim, self.agent.id)
        obs = self._get_observation()
        info = {}
        return obs, info
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.step_cont_action_space(action) if isinstance(self.action_space, spaces.Box) else self.step_discrete_action_space(action)

    def step_cont_action_space(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:

        situational_awareness_build(self.sim, self.agent.id)
        situation = sim_get_situational_awareness(self.sim, self.agent.id)
        ongoing_procedure = sim_get_ongoing_procedure(self.sim, self.agent.id)
        status = PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED

        # Action factors:
        no_op_proc_logit = action[0]
        merge_proc_logit = action[1]
        adjust_speed_proc_logit = action[2]
        stop_proc_logit = action[3]
        cruise_proc_logit = action[4]
        procedure_to_follow = np.argmax([no_op_proc_logit, merge_proc_logit, adjust_speed_proc_logit, stop_proc_logit, cruise_proc_logit])
        current_speed = car_get_speed(self.agent)
        desired_speed = current_speed + (action[5] * from_mph(60))  # Scale action[5] to mph
        top_speed = self.agent.capabilities.top_speed
        desired_speed = np.clip(desired_speed, -from_mph(10), top_speed)  # Clamp desired speed between [-10mph, top_speed_mph]
        duration = (action[6] + 1) * 5.0  # Scale action[6] to [0, 10] seconds
        duration = np.clip(duration, 0.1, 10.0)  # Clamp duration between [0.1s, 10s]
        merge_dir = INDICATOR_LEFT if action[7] < 0 else INDICATOR_RIGHT
        adaptive_cruise = 1.0 if action[8] > 0 else 0.0
        follow_distance_seconds = (action[9] + 1) * 3.0  # Scale action[9] to [0, 6] seconds
        follow_distance_seconds = np.clip(follow_distance_seconds, 0.1, 6.0)  # Clamp follow distance between [0.1s, 6s]
        use_preferred_acceleration_profile = 1.0 if action[10] > 0 else 0.0
        leave_indicators_on = 1.0 if action[11] > 0 else 0.0
        use_linear_braking = 1.0 if action[12] > 0 else 0.0


        if procedure_to_follow == 0:  # NO_OP
            # print(f"Initializing NO_OP procedure with duration {duration} seconds and leave indicators on: {leave_indicators_on}...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_NO_OP, [duration, leave_indicators_on, use_preferred_acceleration_profile])  # Arguments: duration, whether to leave indicators on, whether to use preferred acceleration profile (0 or 1)
        elif procedure_to_follow == 1:  # MERGE
            if merge_dir == INDICATOR_LEFT and not situation.lane_left:
                # print(f"Car {self.agent.id} cannot merge left, lane does not exist.")
                status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
            elif merge_dir == INDICATOR_RIGHT and not situation.lane_right:
                # print(f"Car {self.agent.id} cannot merge right, lane does not exist.")
                status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
            else:
                if current_speed < 0:
                    # print(f"Car {self.agent.id} cannot merge, speed is negative.")
                    status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
                else:
                    # print(f"Initializing MERGE procedure with direction {merge_dir}...")
                    status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_MERGE, [float(merge_dir), duration, max(desired_speed, from_mph(1)), adaptive_cruise, follow_distance_seconds, 0.0])
        elif procedure_to_follow == 2:  # ADJUST_SPEED
            # print(f"Initializing ADJUST_SPEED procedure with speed delta {to_mph(desired_speed)} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_ADJUST_SPEED, [desired_speed, from_mph(0.5), duration, 0.0])
        elif procedure_to_follow == 3:  # STOP
            # print(f"Initializing STOP procedure with linear braking={use_linear_braking} and duration={duration} seconds...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_STOP, [use_linear_braking, duration, use_preferred_acceleration_profile])
        elif procedure_to_follow == 4:  # CRUISE
            # print(f"Initializing CRUISE procedure with adaptive={adaptive_cruise} and speed={to_mph(desired_speed)} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_CRUISE, [duration, max(desired_speed, from_mph(0)), adaptive_cruise, follow_distance_seconds, 0.0])
        else:
            # print(f"Invalid procedure_to_follow: {procedure_to_follow}")
            status = PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED

        reward = 0
        terminated = False
        truncated = False

        if status == PROCEDURE_STATUS_INITIALIZED:
            while (status in [PROCEDURE_STATUS_INITIALIZED, PROCEDURE_STATUS_IN_PROGRESS] and not terminated and not truncated):
                status = procedure_step(self.sim, self.agent, ongoing_procedure)
                if status not in [PROCEDURE_STATUS_IN_PROGRESS, PROCEDURE_STATUS_COMPLETED]:
                    # print(f"Agent {self.agent.id} failed to execute procedure. Status: {status}")
                    break   # Note: Even though procedure_step will recommend some controls even when the procedure already succeeded, we do not apply it since we want the agent to be able to apply new procedures right away.
                simulate(self.sim, self.decision_interval)
                situational_awareness_build(self.sim, self.agent.id)
                reward += self._get_reward() * self.decision_interval  # Accumulate reward over the decision interval
                terminated = self._should_terminate()
                truncated = self._should_truncate()

        else:
            # print(f"Agent {self.agent.id} failed to initialize procedure. Status: {status}")
            car_reset_all_control_variables(self.agent)
            simulate(self.sim, self.decision_interval)  # simulate the step even if procedure initialization failed
            situational_awareness_build(self.sim, self.agent.id)
            reward += self._get_reward() * self.decision_interval  # Accumulate reward over the decision interval
            terminated = self._should_terminate()
            truncated = self._should_truncate()

        obs = self._get_observation()
        info = {"status": status, "time": sim_get_time(self.sim)}
        return obs, reward, terminated, truncated, info

    def step_discrete_action_space(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        situational_awareness_build(self.sim, self.agent.id)
        situation = sim_get_situational_awareness(self.sim, self.agent.id)
        ongoing_procedure = sim_get_ongoing_procedure(self.sim, self.agent.id)
        status = PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED
        if action == 0 or action == 1:
            speed_delta_mag = from_mph(10) if action == 0 else from_mph(5)  # Increase speed by 10 mph or decrease by 5 mph
            speed_delta_direction = 1 if action == 0 else -1
            speed_delta = speed_delta_direction * speed_delta_mag
            # print(f"Initializing ADJUST_SPEED procedure with speed delta {to_mph(speed_delta)} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_ADJUST_SPEED, [car_get_speed(self.agent) + speed_delta, from_mph(0.5), 5.0, 0.0])  # Arguments: desired speed, tolerance, timeout duration, whether to use preferred acceleration profile (0 or 1)
        elif action == 2 or action == 3:
            merge_dir = INDICATOR_LEFT if action == 2 else INDICATOR_RIGHT
            if merge_dir == INDICATOR_LEFT and not situation.lane_left:
                # print(f"Car {self.agent.id} cannot merge left, lane does not exist.")
                status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
            elif merge_dir == INDICATOR_RIGHT and not situation.lane_right:
                # print(f"Car {self.agent.id} cannot merge right, lane does not exist.")
                status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
            else:
                speed = car_get_speed(self.agent)
                if speed < 0:
                    # print(f"Car {self.agent.id} cannot merge, speed is negative.")
                    status = PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE
                else:
                # print(f"Initializing MERGE procedure with direction {merge_dir}...")
                    status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_MERGE, [float(merge_dir), 5.0, max(speed, from_mph(1)), 0.0, 2.0, 0.0])  # Arguments: CarIndicator direction (left/right), timeout_duration, desired cruise speed, whether cruise is adaptive, follow distance in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)
        elif action == 4 or action == 5:
            speed = car_get_speed(self.agent)
            adaptive = 1.0 if action == 5 else 0.0
            # print(f"Initializing CRUISE procedure with adaptive={adaptive} and speed={to_mph(speed)} mph...")
            status = procedure_init(self.sim, self.agent, ongoing_procedure, PROCEDURE_CRUISE, [5.0, max(speed, from_mph(0)), adaptive, 2.0, 0.0])  # Arguments: duration, desired cruise speed, whether adaptive, follow duration in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)

        reward = 0
        terminated = False
        truncated = False

        if status == PROCEDURE_STATUS_INITIALIZED:
            while (status in [PROCEDURE_STATUS_INITIALIZED, PROCEDURE_STATUS_IN_PROGRESS] and not terminated and not truncated):
                status = procedure_step(self.sim, self.agent, ongoing_procedure)
                if status not in [PROCEDURE_STATUS_IN_PROGRESS, PROCEDURE_STATUS_COMPLETED]:
                    # print(f"Agent {self.agent.id} failed to execute procedure. Status: {status}")
                    break   # Note: Even though procedure_step will recommend some controls even when the procedure already succeeded, we do not apply it since we want the agent to be able to apply new procedures right away.
                simulate(self.sim, self.decision_interval)
                situational_awareness_build(self.sim, self.agent.id)
                reward += self._get_reward() * self.decision_interval  # Accumulate reward over the decision interval
                terminated = self._should_terminate()
                truncated = self._should_truncate()

        else:
            # print(f"Agent {self.agent.id} failed to initialize procedure. Status: {status}")
            car_reset_all_control_variables(self.agent)
            simulate(self.sim, self.decision_interval)  # simulate the step even if procedure initialization failed
            situational_awareness_build(self.sim, self.agent.id)
            reward += self._get_reward() * self.decision_interval  # Accumulate reward over the decision interval
            terminated = self._should_terminate()
            truncated = self._should_truncate()

        obs = self._get_observation()
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
        sim_duration=60.0 * 10,
        synchronized=True,  # Set to True if you want synchronized simulation
        should_render=True  # Set to True if you want to connect to the render server
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
