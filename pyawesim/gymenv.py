from typing import Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bindings as A
from gymenv_utils import direction_onehot, indicator_onehot


class AwesimEnv(gym.Env):
    def __init__(self, city_width: int = 1000, goal_lane: Optional[int] = None, num_cars: int = 256, decision_interval: float = 0.1, sim_duration: float = 60 * 60, synchronized: bool = False, synchronized_sim_speedup=1.0, should_render: bool = False, render_server_ip="127.0.0.1", i=0) -> None:
        super().__init__()
        self.goal_lane = goal_lane
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.sim_duration = sim_duration
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render
        self.render_server_ip = render_server_ip

        self.sim = A.sim_malloc()
        A.awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY)
        A.sim_set_agent_enabled(self.sim, True)
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent = A.sim_get_agent_car(self.sim)
        print(f"Agent car ID: {self.agent.id}")

        A.seed_rng(i + 1)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        n_state_factors = len(self._get_observation())
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(n_state_factors,), dtype=np.float32)

    def _construct_car_state_observation(self):
        agent = self.agent
        lane_indicator = indicator_onehot(A.car_get_indicator_lane(agent))
        turn_indicator = indicator_onehot(A.car_get_indicator_turn(agent))
        return np.concatenate((
            np.array([
                agent.capabilities.top_speed / 10,
                agent.capabilities.accel_profile.max_acceleration / 10,
                agent.capabilities.accel_profile.max_acceleration_reverse_gear / 10,
                agent.capabilities.accel_profile.max_deceleration / 10,
                A.car_get_speed(agent) / 10,  # speed in m/s, normalized
                A.car_get_acceleration(agent) / 10,  # acceleration in m/s^2, normalized
                A.car_get_request_indicated_lane(agent),
                A.car_get_request_indicated_turn(agent)
            ], dtype=np.float32),
            lane_indicator,
            turn_indicator
        ))

    def _construct_location_observation(self):
        """Pinpoints the exact lane using meaningful attributes, and the vehicle's position within it."""
        map = A.sim_get_map(self.sim)
        sit = A.sim_get_situational_awareness(self.sim, 0)
        is_on_intersection = sit.is_on_intersection
        center = sit.intersection.center if is_on_intersection else sit.road.mid_point
        center_x, center_y = center.x, center.y
        lane = sit.lane
        direction = lane.direction
        if lane.type == A.LINEAR_LANE:
            direction_from = direction_onehot(direction)
            direction_to = direction_from
        else:
            lane_prev = A.lane_get_connection_incoming_straight(lane, map)
            lane_next = A.lane_get_connection_straight(lane, map)
            direction_from = direction_onehot(lane_prev.direction)
            direction_to = direction_onehot(lane_next.direction)
        is_straight = lane.type == A.LINEAR_LANE
        is_left_turn = lane.direction == A.DIRECTION_CCW
        is_right_turn = lane.direction == A.DIRECTION_CW
        exit_available = sit.is_exit_available  # whether rightmost lane of this road provides opportunity to exit
        merge_available = sit.is_on_entry_ramp  # whether leftmost lane of this road provides opportunity to merge
        approaching_dead_end = sit.is_approaching_dead_end
        approaching_intersection = sit.is_an_intersection_upcoming
        if is_on_intersection:
            if lane.type == A.LINEAR_LANE:  # straight through lane. Number of parallel lanes depends on the roads it connects
                prev_lane = A.lane_get_connection_incoming_straight(lane, map)
                prev_road = A.lane_get_road(prev_lane, map)
                num_lanes = prev_road.num_lanes
                lane_number = A.road_find_index_of_lane(prev_road, prev_lane.id)
            else:   # turn lane. Doesn't have any parallel lanes
                num_lanes = 1
                lane_number = 1
        else:
            num_lanes = sit.road.num_lanes
            lane_number = A.road_find_index_of_lane(sit.road, lane.id)
        lane_number_normalized_from_left = lane_number / num_lanes
        lane_number_normalized_from_right = (num_lanes - 1 - lane_number) / num_lanes
        progress = A.car_get_lane_progress(self.agent)
        distance_to_end_of_lane_from_leading_edge = sit.distance_to_end_of_lane_from_leading_edge  # accounts for the car's length. More meaningful than pointing to the center of the car.
        return np.concatenate((
            np.array([is_on_intersection, center_x / 1000, center_y / 1000, is_straight, is_left_turn, is_right_turn, exit_available, merge_available, approaching_dead_end, approaching_intersection, num_lanes / 10, lane_number_normalized_from_left, lane_number_normalized_from_right, progress, distance_to_end_of_lane_from_leading_edge / 100], dtype=np.float32),
            direction_from, direction_to
        ))

    def _construct_driving_assistant_observation(self):
        das = A.sim_get_driving_assistant(self.sim, 0)
        speed_target = das.speed_target
        pd_mode = das.PD_mode
        position_error = das.position_error
        das_merge_intent = indicator_onehot(das.merge_intent)
        das_turn_intent = indicator_onehot(das.turn_intent)
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
        use_linear_speed_control = das.use_linear_speed_control
        return np.concatenate((
            np.array([speed_target / 10, pd_mode, position_error / 100, follow_mode, follow_mode_thw / 10, follow_mode_buffer, merge_assistance, aeb_assistance, merge_min_thw / 10, aeb_min_thw, feasible_thw / 10, aeb_in_progress, aeb_manually_disengaged, use_preferred_acc_profile, use_linear_speed_control], dtype=np.float32),
            das_merge_intent,
            das_turn_intent
        ))

    def _construct_nearby_vehicles_observation(self):
        sim = self.sim
        agent = self.agent
        speed = A.car_get_speed(agent)
        acc = A.car_get_acceleration(agent)
        situation = A.sim_get_situational_awareness(sim, 0)

        lead_exists = situation.is_vehicle_ahead
        lead_distance = situation.distance_to_lead_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.lead_vehicle) / 2) if lead_exists else 1000
        lead_rel_speed = A.car_get_speed(situation.lead_vehicle) - speed if lead_exists else 0
        lead_rel_acc = A.car_get_acceleration(situation.lead_vehicle) - acc if lead_exists else 0
        behind_exists = situation.is_vehicle_behind
        behind_distance = situation.distance_to_following_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.following_vehicle) / 2) if behind_exists else 1000
        behind_rel_speed = A.car_get_speed(situation.following_vehicle) - speed if behind_exists else 0
        behind_rel_acc = A.car_get_acceleration(situation.following_vehicle) - acc if behind_exists else 0

        nearby_flattened = A.NearbyVehiclesFlattened()
        A.nearby_vehicles_flatten(
            situation.nearby_vehicles, nearby_flattened, True)
        flattened_count = A.nearby_vehicles_flattened_get_count(nearby_flattened)
        nearby_exists_flags = np.array([A.nearby_vehicles_flattened_get_car_id(
            nearby_flattened, i) != -1 for i in range(flattened_count)], dtype=np.float32)
        nearby_distances = np.array([A.nearby_vehicles_flattened_get_distance(nearby_flattened, i) for i in range(
            flattened_count)], dtype=np.float32).clip(0, 1000.0)  # Clip distances to [0, 1000] meters
        nearby_ttcs = np.array([A.nearby_vehicles_flattened_get_time_to_collision(nearby_flattened, i) for i in range(
            flattened_count)], dtype=np.float32).clip(0, 10.0)  # Clip TTC to [0, 10] seconds
        nearby_thws = np.array([A.nearby_vehicles_flattened_get_time_headway(nearby_flattened, i) for i in range(
            flattened_count)], dtype=np.float32).clip(0, 10.0)  # Clip THW to [0, 10] seconds
        nearby_speeds_relative = np.array([(A.car_get_speed(A.sim_get_car(sim, A.nearby_vehicles_flattened_get_car_id(
            nearby_flattened, i))) - speed) if nearby_exists_flags[i] else 0 for i in range(flattened_count)], dtype=np.float32)
        nearby_acc_relative = np.array([(A.car_get_acceleration(A.sim_get_car(sim, A.nearby_vehicles_flattened_get_car_id(
            nearby_flattened, i))) - acc) if nearby_exists_flags[i] else 0 for i in range(flattened_count)], dtype=np.float32)
        return np.concatenate((np.array([lead_exists, lead_distance / 100, lead_rel_speed / 10, lead_rel_acc / 10, behind_exists, behind_distance / 100, behind_rel_speed / 10, behind_rel_acc / 10]), nearby_distances / 100, nearby_ttcs / 10, nearby_thws / 10, nearby_speeds_relative / 10, nearby_acc_relative / 10))
        # todo: observe attributes like indicators and braking capabilities of the nearby vehicles

    def _construct_environment_conditions_observation(self):
        sit = A.sim_get_situational_awareness(self.sim, 0)
        speed_limit = sit.lane.speed_limit
        return np.array([speed_limit / 10], dtype=np.float32)

    def _construct_turn_relevant_observations(self):
        sit = A.sim_get_situational_awareness(self.sim, 0)
        lane = sit.lane
        map = A.sim_get_map(self.sim)
        left_turn_available = A.lane_get_connection_left_id(lane) != A.ID_NULL
        straight_available = A.lane_get_connection_straight_id(lane) != A.ID_NULL
        right_turn_available = A.lane_get_connection_right_id(lane) != A.ID_NULL
        is_green = False
        is_yellow = False
        is_red = False
        all_red = False
        is_fourway_stop = False
        countdown = 0
        intersection = sit.intersection
        if intersection:
            is_fourway_stop = intersection.state == A.FOUR_WAY_STOP
            if not is_fourway_stop:
                direction = A.lane_get_connection_incoming_straight(lane, map).direction if sit.is_on_intersection else lane.direction
                is_EW = direction in (A.DIRECTION_EAST, A.DIRECTION_WEST)
                is_NS = not is_EW
                is_green = (is_EW and intersection.state == A.NS_RED_EW_GREEN) or (is_NS and intersection.state == A.NS_GREEN_EW_RED)
                is_yellow = (is_EW and intersection.state == A.EW_YELLOW_NS_RED) or (is_NS and intersection.state == A.NS_YELLOW_EW_RED)
                is_red = not is_green and not is_yellow
                all_red = intersection.state in (A.ALL_RED_BEFORE_EW_GREEN, A.ALL_RED_BEFORE_NS_GREEN)
                countdown = intersection.countdown
            else:
                all_red = True
                is_red = True

        return np.array([left_turn_available, straight_available, right_turn_available, is_green, is_yellow, is_red, all_red, is_fourway_stop, countdown / 60], dtype=np.float32)

    def _get_observation(self):
        A.situational_awareness_build(self.sim, self.agent.id)
        return np.concatenate((
            self._construct_car_state_observation(),
            self._construct_driving_assistant_observation(),
            self._construct_location_observation(),
            self._construct_nearby_vehicles_observation(),
            self._construct_turn_relevant_observations(),
            self._construct_environment_conditions_observation(),
        ))

    def _power_applied_by_wheels_to_accelerate(self):
        # Calculate the power applied by all wheels based on the current speed and acceleration
        my_speed = A.car_get_speed(self.agent)
        acceleration = A.car_get_acceleration(self.agent)
        if (my_speed > 0 and my_speed * acceleration > 0):  # speeding up
            return my_speed * acceleration  # W/Kg
        return 0    # braking or reversing

    def _power_applied_by_wheels_to_reverse(self):
        # Calculate the power applied by all wheels based on the current speed and acceleration
        my_speed = A.car_get_speed(self.agent)
        acceleration = A.car_get_acceleration(self.agent)
        if (my_speed < 0 and my_speed * acceleration > 0):  # reversing
            return my_speed * acceleration  # W/Kg
        return 0    # braking or reversing

    def _power_applied_by_brakes(self):
        # Calculate the power applied by the brakes based on the current speed and braking force
        my_speed = A.car_get_speed(self.agent)
        acceleration = A.car_get_acceleration(self.agent)
        if (my_speed * acceleration < 0):  # braking
            return -my_speed * acceleration  # W/Kg
        return 0

    def _get_reward(self):
        # encourage moving forward
        return A.to_mph(A.car_get_speed(self.agent)) / 100

    def _crashed(self):
        # detect crash with lead vehicle
        situation = A.sim_get_situational_awareness(self.sim, 0)
        if situation.is_vehicle_ahead:
            distance_center_to_center = situation.distance_to_lead_vehicle
            distance_bumper_to_tail = distance_center_to_center - \
                (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.lead_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                # print(f"Agent {self.agent.id} crashed into lead vehicle #{situation.lead_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        # detect crash with following vehicle
        if situation.is_vehicle_behind:
            distance_center_to_center = situation.distance_to_following_vehicle
            distance_bumper_to_tail = distance_center_to_center - (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.following_vehicle) / 2)
            if distance_bumper_to_tail <= 0:
                # print(f"Agent {self.agent.id} crashed into following vehicle #{situation.following_vehicle.id} at distance {distance_bumper_to_tail}.")
                return True
        return False

    def _reached_destination(self):
        if self.goal_lane is None:
            return False
        progress = A.car_get_lane_progress(self.agent)
        return A.car_get_lane_id(self.agent) == 84 and 0.4 < progress and progress < 0.6  # Lane 84 is the destination lane in the canonical map and you have to get to the middle of it

    def _should_terminate(self):
        return self._crashed() or self._reached_destination()

    def _should_truncate(self):
        # truncate if simulation time exceeds sim_duration
        return A.sim_get_time(self.sim) >= self.sim_duration

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        A.sim_disconnect_from_render_server(self.sim)
        if self.goal_lane is None:
            A.awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY)
        else:
            first_reset_done = False
            while not first_reset_done or A.car_get_lane_id(A.sim_get_car(self.sim, 0)) == self.goal_lane:
                A.awesim_setup(self.sim, self.city_width, self.num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY)
                first_reset_done = True

        if self.synchronized:
            A.sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            A.sim_connect_to_render_server(self.sim, self.render_server_ip, 4242)

        A.sim_set_agent_enabled(self.sim, True)
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent = A.sim_get_agent_car(self.sim)
        self.das = A.sim_get_driving_assistant(self.sim, self.agent.id)
        A.driving_assistant_reset_settings(self.das, self.agent)

        # ----------- AEB enabled with hardcoded parameters ---------------------------- TODO: Make this adjustable in action space.
        A.driving_assistant_configure_aeb_assistance(
            self.das, self.agent, self.sim, True)
        A.driving_assistant_configure_aeb_min_thw(
            self.das, self.agent, self.sim, 0.5)
        # ------------------------------------------------------------------------------

        A.situational_awareness_build(self.sim, self.agent.id)
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # configure das based on action

        # print(f"Action received: {action}")

        a_id = 0

        # speed adjustment. +/- 10 mps
        speed_adjustment = float(action[a_id]) * A.mps(10)
        speed_target_new = self.das.speed_target + speed_adjustment
        speed_target_new = np.clip(
            speed_target_new, A.from_mph(-10), self.agent.capabilities.top_speed)
        # print(f"Setting speed target to {to_mph(speed_target_new)} mph (adjusted by {to_mph(speed_adjustment)} mph)")
        A.driving_assistant_configure_speed_target(
            self.das, self.agent, self.sim, A.mps(speed_target_new))
        a_id += 1

        # PD Switch
        pd_prob = (action[a_id] + 1) / 2
        pd_mode = self.np_random.random() < pd_prob
        A.driving_assistant_configure_PD_mode(self.das, self.agent, self.sim, bool(pd_mode))
        a_id += 1

        # position error adjustment
        if pd_mode:
            position_error_adjustment = float(action[a_id]) * A.meters(50)  # Adjust by +/- 50 meters
            new_position_error = self.das.position_error + position_error_adjustment
            # print(f"Setting position error to {new_position_error} m")
            A.driving_assistant_configure_position_error(self.das, self.agent, self.sim, A.meters(new_position_error))
        a_id += 1

        # merge intent adjustment.
        should_indicate = self.np_random.random() < abs(action[a_id])
        new_merge_intent = (
            A.INDICATOR_LEFT if action[a_id] < 0 else A.INDICATOR_RIGHT) if should_indicate else A.INDICATOR_NONE
        # print(f"Setting merge intent to {new_merge_intent}")
        A.driving_assistant_configure_merge_intent(
            self.das, self.agent, self.sim, new_merge_intent)
        a_id += 1

        # turn intent adjustment:
        should_indicate = self.np_random.random() < abs(action[a_id])
        new_turn_intent = (
            A.INDICATOR_LEFT if action[a_id] < 0 else A.INDICATOR_RIGHT) if should_indicate else A.INDICATOR_NONE
        # print(f"Setting turn intent to {new_turn_intent}")
        A.driving_assistant_configure_turn_intent(
            self.das, self.agent, self.sim, new_turn_intent)
        a_id += 1

        # Follow mode switch
        follow_prob = (action[a_id] + 1) / 2
        follow_mode = self.np_random.random() < follow_prob
        if follow_mode and speed_target_new < 0:
            speed_target_new = 0  # Follow mode requires positive speed
            A.driving_assistant_configure_speed_target(self.das, self.agent, self.sim, A.mps(speed_target_new))
        A.driving_assistant_configure_follow_mode(self.das, self.agent, self.sim, bool(follow_mode))
        a_id += 1

        # Follow mode THW adjustment
        follow_thw_adjustment = float(action[a_id]) * A.seconds(1)  # Adjust by +/- 1 second
        new_follow_thw = self.das.follow_mode_thw + follow_thw_adjustment
        new_follow_thw = np.clip(new_follow_thw, 0.2, 5.0)
        A.driving_assistant_configure_follow_mode_thw(self.das, self.agent, self.sim, A.seconds(new_follow_thw))
        a_id += 1

        # use linear speed control switch
        use_linear_speed_control_prob = (action[a_id] + 1) / 2
        use_linear_speed_control = self.np_random.random() < use_linear_speed_control_prob
        A.driving_assistant_configure_use_linear_speed_control(self.das, self.agent, self.sim, bool(use_linear_speed_control))
        a_id += 1

        # TODO: control other DAS parameters

        # simulate for a while. das will operate for us during this time
        done = False
        crashed = False
        reached_goal = False
        t = 0
        if self.goal_lane is not None:
            r = 0  # time loss and heat loss penalty
        else:
            r = 0  # distance moved forward minus heat loss penalty
        # Check for termination conditions every 0.1 seconds
        termination_check_interval = 0.1
        while not done and t < self.decision_interval:
            # We are breaking the decision_interval into smaller steps to check for termination or truncation conditions more frequently
            A.simulate(self.sim, termination_check_interval)
            t += termination_check_interval
            # Update situational awareness after simulation step
            A.situational_awareness_build(self.sim, self.agent.id)
            if self.goal_lane is None:
                # integrate speed over time to get displacement
                r += self._get_reward() * termination_check_interval
            else:
                r -= termination_check_interval  # time penalty
            # integrate braking power over time to get energy lost
            r -= 0.01 * self._power_applied_by_brakes() * termination_check_interval
            crashed = self._crashed()
            reached_goal = self._reached_destination()
            done = crashed or reached_goal or self._should_truncate()

        # crashed = self._crashed()
        # if self._crashed():
        #     r -= 0.5 * car_get_speed(self.agent) ** 2   # All kinetic energy lost

        reward = r
        terminated = crashed or reached_goal
        truncated = self._should_truncate()
        obs = self._get_observation()

        obs = self._get_observation()
        info = {"time": A.sim_get_time(self.sim)}
        info["is_success"] = not crashed if self.goal_lane is None else reached_goal
        info["crashed"] = crashed
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        A.sim_disconnect_from_render_server(self.sim)
        A.sim_free(self.sim)
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
        should_render=False  # Set to True if you want to connect to the render server
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

            if terminated or truncated:
                print(f"Episode finished after {step_count} steps")
                print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        env.close()
        print("Environment closed")
