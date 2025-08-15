from typing import Any, SupportsFloat, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bindings import *

class AwesimEnv(gym.Env):
    def __init__(self, city_width: int = 1000, num_cars: int = 256, decision_interval: float = 0.1, sim_duration: float = 60 * 60, synchronized: bool = False, synchronized_sim_speedup = 1.0, should_render: bool = False, i = 0) -> None:
        super().__init__()
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.sim_duration = sim_duration
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render

        self.sim = sim_malloc()
        seed_rng(i + 1)
        sim_init(self.sim)
        sim_set_agent_enabled(self.sim, True)
        sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent: Car = sim_get_new_car(self.sim)
        self.das: DrivingAssistant = sim_get_driving_assistant(self.sim, self.agent.id)
        print(f"Agent car ID: {self.agent.id}")


        # action factors to adjust driving assistant state:
        # 0. speed_target_adjustment: -1: subtract 10 mph, 1: add 60 mph.
        # 1. PD toggle
        # 2. position_error_adjustment
        # 3. merge_intent adjustment
        # 4. turn_intent adjustment
        # 5. cruise_mode toggle
        # 6. cruise_speed_adjustment
        # 7. follow_mode toggle
        # 8. follow_mode_thw_adjustment
        # 9. follow_mode_buffer_adjustment
        # 10. merge_assistance toggle
        # 11. aeb_assistance toggle
        # 12. merge_min_thw_adjustment
        # 13. aeb_min_thw_adjustment
        # 14. aeb manually disengage
        # 15. use_preferred_acceleration_profile toggle

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        n_state_factors = len(self._get_observation())
        self.observation_space = spaces.Box(low=-10, high=10, shape=(n_state_factors,), dtype=np.float32)

    def _get_observation(self):
        agent = self.agent
        sim = self.sim
        situation = sim_get_situational_awareness(sim, 0)
        das = sim_get_driving_assistant(sim, 0)

        # driving stuff:
        speed = car_get_speed(agent)
        acc = car_get_acceleration(agent)
        speed_target = das.speed_target
        pd_mode = das.PD_mode
        position_error = das.position_error
        das_merge_intent = das.merge_intent - 1
        das_turn_intent = das.turn_intent - 1
        follow_mode = das.follow_mode
        follow_mode_thw = das.follow_mode_thw
        follow_mode_buffer = das.follow_mode_buffer
        merge_assistance = das.merge_assistance
        aeb_assistance = das.aeb_assistance
        merge_min_thw = das.merge_min_thw
        aeb_min_thw = das.aeb_min_thw
        feasible_thw = np.clip(das.feasible_thw, 0.0, 100.0)
        aeb_in_progress = das.aeb_in_progress
        aeb_manually_disengaged = das.aeb_manually_disengaged
        use_preferred_acc_profile = das.use_preferred_accel_profile

        # traffic around us:
        nearby_flattened = NearbyVehiclesFlattened()
        nearby_vehicles_flatten(situation.nearby_vehicles, nearby_flattened, True)
        flattened_count = nearby_vehicles_flattened_get_count(nearby_flattened)
        nearby_exists_flags = np.array([nearby_vehicles_flattened_get_car_id(nearby_flattened, i) != -1 for i in range(flattened_count)], dtype=np.float32)
        nearby_distances = np.array([nearby_vehicles_flattened_get_distance(nearby_flattened, i) for i in range(flattened_count)], dtype=np.float32).clip(0, 1000.0)  # Clip distances to [0, 1000] meters
        nearby_ttcs = np.array([nearby_vehicles_flattened_get_time_to_collision(nearby_flattened, i) for i in range(flattened_count)], dtype=np.float32).clip(0, 10.0)  # Clip TTC to [0, 10] seconds
        nearby_thws = np.array([nearby_vehicles_flattened_get_time_headway(nearby_flattened, i) for i in range(flattened_count)], dtype=np.float32).clip(0, 10.0)  # Clip THW to [0, 10] seconds
        nearby_speeds_relative = np.array([(car_get_speed(sim_get_car(sim, nearby_vehicles_flattened_get_car_id(nearby_flattened, i))) - speed) if nearby_exists_flags[i] else 0 for i in range(flattened_count)], dtype=np.float32)
        nearby_acc_relative = np.array([(car_get_acceleration(sim_get_car(sim, nearby_vehicles_flattened_get_car_id(nearby_flattened, i))) - car_get_acceleration(agent)) if nearby_exists_flags[i] else 0 for i in range(flattened_count)], dtype=np.float32)

        # map variables:
        lane_indicator = car_get_indicator_lane(agent) - 1
        turn_indicator = car_get_indicator_turn(agent) - 1
        lane_change_requested = car_get_request_indicated_lane(agent)
        turn_requested = car_get_request_indicated_turn(agent)
        is_on_leftmost_lane = situation.is_on_leftmost_lane
        is_on_rightmost_lane = situation.is_on_rightmost_lane
        left_exists = situation.is_lane_change_left_possible    # merge considered
        right_exists = situation.is_lane_change_right_possible  # exit considered
        exit_available = situation.is_exit_available # if rightmost lane of this road provides opportunity to exit
        merge_available = situation.is_on_entry_ramp # if leftmost lane of this road provides opportunity to merge
        approaching_dead_end = situation.is_approaching_dead_end
        approaching_intersection = situation.is_an_intersection_upcoming
        distance_to_lane_end = situation.distance_to_end_of_lane_from_leading_edge
        # is_turn_left_possible = # TODO
        # is_turn_right_possible = # TODO

        # # concat all these into a float32 np array
        obs = np.concatenate([
            # Driving assistant state
            np.array([
            speed / 10,
            acc,
            speed_target / 10,
            pd_mode,
            position_error / 100,
            das_merge_intent,
            # das_turn_intent,
            # follow_mode,
            # follow_mode_thw / 10,
            # follow_mode_buffer / 10,
            # merge_assistance,
            # aeb_assistance,
            # merge_min_thw / 10,
            # aeb_min_thw / 10,
            feasible_thw / 10,
            aeb_in_progress,
            # aeb_manually_disengaged,
            # use_preferred_acc_profile
            ], dtype=np.float32),

            # Nearby vehicles existence flags
            nearby_exists_flags.astype(np.float32),

            # Nearby vehicles distances
            nearby_distances.astype(np.float32) / 100,

            # Nearby vehicles relative speeds
            nearby_speeds_relative.astype(np.float32) / 10,

            # Nearby vehicles relative accelerations
            nearby_acc_relative.astype(np.float32),

            # Nearby vehicles time-to-collision
            nearby_ttcs.astype(np.float32) / 10,

            # Nearby vehicles time-headway
            nearby_thws.astype(np.float32) / 10,

            # Map and situational variables
            np.array([
            lane_indicator,
            # turn_indicator,
            # lane_change_requested,
            # turn_requested,
            is_on_leftmost_lane,
            is_on_rightmost_lane,
            left_exists,
            right_exists,
            exit_available,
            merge_available,
            approaching_dead_end,
            approaching_intersection,
            distance_to_lane_end / 100
            ], dtype=np.float32)
        ])

        return obs

    def _power_applied_by_wheels_to_accelerate(self):
        # Calculate the power applied by all wheels based on the current speed and acceleration
        situation = sim_get_situational_awareness(self.sim, 0)
        my_speed = car_get_speed(self.agent)
        acceleration = car_get_acceleration(self.agent)
        if (my_speed > 0 and my_speed * acceleration > 0): # speeding up
            return my_speed * acceleration  # W/Kg
        return 0    # braking or reversing

    def _power_applied_by_wheels_to_reverse(self):
        # Calculate the power applied by all wheels based on the current speed and acceleration
        situation = sim_get_situational_awareness(self.sim, 0)
        my_speed = car_get_speed(self.agent)
        acceleration = car_get_acceleration(self.agent)
        if (my_speed < 0 and my_speed * acceleration > 0): # reversing
            return my_speed * acceleration  # W/Kg
        return 0    # braking or reversing

    def _power_applied_by_brakes(self):
        # Calculate the power applied by the brakes based on the current speed and braking force
        situation = sim_get_situational_awareness(self.sim, 0)
        my_speed = car_get_speed(self.agent)
        acceleration = car_get_acceleration(self.agent)
        if (my_speed * acceleration < 0): # braking
            return -my_speed * acceleration  # W/Kg
        return 0

    def _get_reward(self):
        # encourage moving forward
        return to_mph(car_get_speed(self.agent)) / 100

    def _crashed(self):
        # detect crash with lead vehicle
        situation = sim_get_situational_awareness(self.sim, 0)
        if situation.is_vehicle_ahead:
            distance_center_to_center = situation.distance_to_lead_vehicle
            distance_bumper_to_tail = distance_center_to_center - (car_get_length(self.agent) / 2 + car_get_length(situation.lead_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                # print(f"Agent {self.agent.id} crashed into lead vehicle #{situation.lead_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        # detect crash with following vehicle
        if situation.is_vehicle_behind:
            distance_center_to_center = situation.distance_to_following_vehicle
            distance_bumper_to_tail = distance_center_to_center - (car_get_length(self.agent) / 2 + car_get_length(situation.following_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                # print(f"Agent {self.agent.id} crashed into following vehicle #{situation.following_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        return False

    def _should_terminate(self):
        return self._crashed()

    def _should_truncate(self):
        # truncate if simulation time exceeds sim_duration
        return sim_get_time(self.sim) >= self.sim_duration

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        sim_disconnect_from_render_server(self.sim)
        awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, clock_reading(0, 8, 0, 0), WEATHER_SUNNY)
        if self.synchronized:
            sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            sim_connect_to_render_server(self.sim, "127.0.0.1", 4242)

        sim_set_agent_enabled(self.sim, True)
        sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent = sim_get_agent_car(self.sim)
        self.das = sim_get_driving_assistant(self.sim, self.agent.id)
        driving_assistant_reset_settings(self.das, self.agent)

        # ----------- AEB enabled with hardcoded parameters ---------------------------- TODO: Make this adjustable in action space.
        driving_assistant_configure_aeb_assistance(self.das, self.agent, self.sim, True)
        driving_assistant_configure_aeb_min_thw(self.das, self.agent, self.sim, 0.5)
        # ------------------------------------------------------------------------------

        situational_awareness_build(self.sim, self.agent.id)
        obs = self._get_observation()
        info = {}
        return obs, info


    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # configure das based on action

        # print(f"Action received: {action}")

        a_id = 0

        # speed adjustment. +/- 10 mph
        speed_adjustment = float(action[a_id]) * from_mph(10)
        speed_target_new = self.das.speed_target + speed_adjustment
        speed_target_new = np.clip(speed_target_new, from_mph(-10), from_mph(self.agent.capabilities.top_speed))
        # print(f"Setting speed target to {to_mph(speed_target_new)} mph (adjusted by {to_mph(speed_adjustment)} mph)")
        driving_assistant_configure_speed_target(self.das, self.agent, self.sim, mps(speed_target_new))
        a_id += 1

        # PD toggle
        # pd_toggle_prob = (action[a_id] + 1) / 2
        # pd_mode = not self.das.PD_mode if self.np_random.random() < pd_toggle_prob else self.das.PD_mode
        # # print(f"Setting PD mode to {'enabled' if pd_mode else 'disabled'}")
        # driving_assistant_configure_PD_mode(self.das, self.agent, self.sim, pd_mode)
        # a_id += 1

        # # position error adjustment
        # if pd_mode:
        #     position_error_adjustment = float(action[a_id]) * meters(10)  # Adjust by +/- 10 meters
        #     new_position_error = self.das.position_error + position_error_adjustment
        #     # print(f"Setting position error to {new_position_error} m")
        #     driving_assistant_configure_position_error(self.das, self.agent, self.sim, meters(new_position_error))
        # a_id += 1

        # merge intent adjustment.
        should_indicate = self.np_random.random() < abs(action[a_id])
        new_merge_intent = (INDICATOR_LEFT if action[a_id] < 0 else INDICATOR_RIGHT) if should_indicate else INDICATOR_NONE
        # print(f"Setting merge intent to {new_merge_intent}")
        driving_assistant_configure_merge_intent(self.das, self.agent, self.sim, new_merge_intent)
        a_id += 1

        # TODO: control other DAS parameters

        # simulate for a while. das will operate for us during this time
        done = False
        t = 0
        r = 0  # distance moved forward minus heat loss penalty
        termination_check_interval = 0.1  # Check for termination conditions every 0.1 seconds
        while not done and t < self.decision_interval:
            # We are breaking the decision_interval into smaller steps to check for termination or truncation conditions more frequently
            simulate(self.sim, termination_check_interval)
            t += termination_check_interval
            situational_awareness_build(self.sim, self.agent.id)    # Update situational awareness after simulation step
            r += self._get_reward() * termination_check_interval    # integrate speed over time to get displacement
            r -= 0.01 * self._power_applied_by_brakes() * termination_check_interval   # integrate braking power over time to get energy lost
            done = self._should_terminate() or self._should_truncate()


        # crashed = self._crashed()
        # if self._crashed():
        #     r -= 0.5 * car_get_speed(self.agent) ** 2   # All kinetic energy lost

        reward = r
        terminated = self._should_terminate()
        truncated = self._should_truncate()
        obs = self._get_observation()
        
        obs = self._get_observation()
        info = {"time": sim_get_time(self.sim)}
        info["is_success"] = not terminated
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
