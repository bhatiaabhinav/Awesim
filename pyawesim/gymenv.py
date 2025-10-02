from typing import Any, Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bindings as A
from gymenv_utils import direction_onehot, indicator_onehot


class AwesimEnv(gym.Env):
    """
    A Gym environment simulating an autonomous vehicle with an Advanced Driver Assistance System (ADAS).
    The agent controls the vehicle to navigate a city environment, aiming to reach a specified goal lane
    or drive safely without crashing. The environment uses a custom simulation backend (`bindings`) and
    provides observations of the vehicle's state, ADAS settings, location, nearby vehicles, and traffic conditions.
    Actions control speed, turn/merge signals, and follow distance. The environment supports rendering and
    synchronized simulation for visualization.

    Key Features:
    - Action space: Continuous 3D Box for speed adjustment (±10 mph), turn/merge signal, and follow distance (±1s).
    - Observation space: 149-dimensional vector capturing vehicle state, ADAS settings, location, nearby vehicles,
      and traffic conditions, normalized to [-10, 10].
    - Reward: -0.2 per second (time penalty), +100 for reaching goal lane, -100 for crashing, and speed-based reward
      if no goal is specified.
    - Termination: Episode ends on crash, reaching goal, or exceeding simulation duration (default 1 hour).
    """

    # Constants
    DEFAULT_CITY_WIDTH = 1000  # City width in meters
    DEFAULT_NUM_CARS = 256  # Number of vehicles in the simulation
    DEFAULT_DECISION_INTERVAL = 0.1  # Time between agent decisions (seconds)
    DEFAULT_SIM_DURATION = 60 * 60  # Simulation duration (seconds, default 1 hour)
    DEFAULT_APPOINTMENT_TIME = 60 * 30  # Default appointment time (seconds, 30 minutes)
    DEFAULT_TIME_PENALTY = 0.0  # Penalty per second for time spent
    DEFAULT_GOAL_REWARD = 1.0  # Reward for reaching the goal lane
    DEFAULT_CRASH_PENALTY = 0.0  # Penalty for crashing
    DEFAULT_RENDER_IP = "127.0.0.1"  # Default IP for rendering server
    TERMINATION_CHECK_INTERVAL = 0.1  # Interval for checking termination conditions (seconds)
    SPEED_ADJUSTMENT_MPH = 10  # Max speed adjustment per action (±10 mph)
    FOLLOW_THW_ADJUSTMENT_SECONDS = 1  # Max follow time-headway adjustment (±1 second)
    MIN_FOLLOW_THW = 0.2  # Minimum follow time-headway (seconds)
    MAX_FOLLOW_THW = 5.0  # Maximum follow time-headway (seconds)
    MAX_DISTANCE = 1000.0  # Maximum distance for nearby vehicles (meters)
    MAX_TTC = 10.0  # Maximum time-to-collision (seconds)
    MAX_THW = 10.0  # Maximum time-headway (seconds)
    COST_GAS_PER_GALLON = 3.2  # Average cost of gas per gallon ($) in US
    COST_LOST_WAGE_PER_HOUR = 30.0  # Average wages per hour ($) in US
    MAY_LOSE_ENTIRE_DAY_WAGE = True  # Lose entire 8-hour workday wage if not reached by the end of sim_duration
    COST_CAR = 35000.0  # Average new car cost ($) in US. This is the cost for total loss in a crash.
    DEFAULT_INIT_FUEL_GALLONS = 15.0  # Default initial fuel level in gallons
    OBSERVATION_NORMALIZATION = {
        "speed": 10,  # Normalize speeds by 10 m/s
        "acceleration": 10,  # Normalize accelerations by 10 m/s²
        "distance": 100,  # Normalize distances by 100 meters
        "center": 1000,  # Normalize coordinates by 1000 meters
        "lanes": 10,  # Normalize lane counts by 10
        "countdown": 60,  # Normalize traffic light countdown by 60 seconds
        'time': 3600,  # Normalize time by 3600 seconds (1 hour)
        'fuel': 50,  # Normalize fuel level by 50 liters (~13.2 gallons)
    }

    def __init__(
        self,
        city_width: int = DEFAULT_CITY_WIDTH,
        goal_lane: Optional[int] = None,
        num_cars: int = DEFAULT_NUM_CARS,
        decision_interval: float = DEFAULT_DECISION_INTERVAL,
        sim_duration: float = DEFAULT_SIM_DURATION,
        appointment_time: float = DEFAULT_APPOINTMENT_TIME,  # time in seconds by which the agent should reach the goal lane, otherwise lost wage cost will be start accumulating
        time_penalty_per_second: float = DEFAULT_TIME_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        crash_penalty: float = DEFAULT_CRASH_PENALTY,
        cost_gas_per_gallon: float = COST_GAS_PER_GALLON,
        cost_lost_wage_per_hour: float = COST_LOST_WAGE_PER_HOUR,
        may_lose_entire_day_wage: bool = MAY_LOSE_ENTIRE_DAY_WAGE,
        cost_car: float = COST_CAR,
        init_fuel_gallons: float = DEFAULT_INIT_FUEL_GALLONS,
        synchronized: bool = False,
        synchronized_sim_speedup: float = 1.0,
        should_render: bool = False,
        render_server_ip: str = DEFAULT_RENDER_IP,
        env_index: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the AwesimEnv with simulation and agent configuration.

        Args:
            city_width (int): Width of the city map in meters.
            goal_lane (Optional[int]): Target lane ID to reach, or None for general driving.
            num_cars (int): Number of vehicles in the simulation.
            decision_interval (float): Time interval between agent actions (seconds).
            sim_duration (float): Maximum duration of an episode (seconds).
            appointment_time (float): Time in seconds by which the agent should reach the goal lane, otherwise lost wage cost will start accumulating.
            time_penalty_per_second (float): Penalty per second for time spent.
            goal_reward (float): Reward for reaching the goal lane.
            crash_penalty (float): Penalty for crashing.
            cost_gas_per_gallon (float): Cost of gas per gallon ($).
            cost_lost_wage_per_hour (float): Cost of lost wages per hour ($).
            may_lose_entire_day_wage (bool): If True, lose entire 8-hour workday wage if not reached by the end of sim_duration.
            cost_car (float): Cost of the car ($), applied on total loss in a crash.
            init_fuel_gallons (float): Initial fuel level in gallons.
            synchronized (bool): If True, synchronize simulation for rendering.
            synchronized_sim_speedup (float): Speedup factor for synchronized simulation.
            should_render (bool): If True, connect to rendering server.
            render_server_ip (str): IP address of the rendering server.
            env_index (int): Index for environment instance (used for seeding).
            verbose (bool): If True, print action and event information.
        """
        super().__init__()
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")

        # Environment configuration
        self.goal_lane = goal_lane
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.time_penalty_per_second = time_penalty_per_second
        self.goal_reward = goal_reward
        self.crash_penalty = crash_penalty
        self.cost_gas_per_gallon = cost_gas_per_gallon
        self.cost_lost_wage_per_hour = cost_lost_wage_per_hour
        self.may_lose_entire_day_wage = may_lose_entire_day_wage
        self.cost_car = cost_car
        self.init_fuel_gallons = init_fuel_gallons
        self.sim_duration = sim_duration
        self.appointment_time = appointment_time
        self.steps = 0
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render
        self.render_server_ip = render_server_ip
        self.verbose = verbose

        # Initialize simulation
        self.sim = A.sim_malloc()  # Allocate simulation instance
        A.awesim_setup(
            self.sim, self.city_width, self.num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY
        )  # Set up city with fixed time (8 AM) and sunny weather
        A.sim_set_agent_enabled(self.sim, True)  # Enable agent control
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)  # Enable ADAS
        self.agent = A.sim_get_agent_car(self.sim)  # Get agent vehicle
        if self.verbose:
            print(f"Agent car ID: {self.agent.id}")

        A.seed_rng(env_index + 1)  # Seed random number generator for reproducibility

        # Define action space. # 5 continuous actions: follow mode vs stop at lane end mode, speed limit, follow mode THW, margin feet (for follow distance buffer (in addition to THW) in follow mode, or stopping margin from end of lane when in stop mode.)
        follow_mode_probability_low, follow_mode_probability_high = 0.0, 1.0        # normalized to [0, 1] where 0 means always in stop mode at lane end, and 1 means always in follow mode, anything in between probabilistically sets follow mode with that probability
        speed_target_low, speed_target_high = 0.0, 1.0                              # normalized to [0, 1] where 0 means 0 mph and 1 means agent's top speed
        follow_mode_thw_low, follow_mode_thw_high = self.MIN_FOLLOW_THW, self.MAX_FOLLOW_THW  # in seconds
        margin_min, margin_max = A.from_feet(1), A.from_feet(10)                    # in feet, margin for follow distance buffer (in addition to THW) in follow mode, or stopping margin from end of lane when in stop mode.
        turn_signal_min, turn_signal_max = -1.0, 1.0                                # -1 means left signal, +1 means right signal, 0 means no signal, anything in between means probabilistically choose left / right with probability proportional to magnitude of the value
        self.action_space = spaces.Box(
            low = np.array([follow_mode_probability_low, speed_target_low, follow_mode_thw_low, margin_min, turn_signal_min], dtype=np.float32),
            high = np.array([follow_mode_probability_high, speed_target_high, follow_mode_thw_high, margin_max, turn_signal_max], dtype=np.float32),
        )

        # Define observation space
        n_state_factors = len(self._get_observation())
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(n_state_factors,), dtype=np.float32
        )  # Normalized observation space

    def _construct_car_state_observation(self) -> np.ndarray:
        """
        Construct observation for the agent's vehicle state.

        Returns:
            np.ndarray: Concatenated array of vehicle capabilities, current speed,
            acceleration, lane/turn indicators, and one-hot encoded signals.
        """
        agent = self.agent
        lane_indicator = indicator_onehot(A.car_get_indicator_lane(agent))
        turn_indicator = indicator_onehot(A.car_get_indicator_turn(agent))
        return np.concatenate(
            (
                np.array(
                    [
                        agent.capabilities.top_speed / self.OBSERVATION_NORMALIZATION["speed"],
                        agent.capabilities.accel_profile.max_acceleration / self.OBSERVATION_NORMALIZATION["acceleration"],
                        agent.capabilities.accel_profile.max_acceleration_reverse_gear / self.OBSERVATION_NORMALIZATION["acceleration"],
                        agent.capabilities.accel_profile.max_deceleration / self.OBSERVATION_NORMALIZATION["acceleration"],
                        A.car_get_speed(agent) / self.OBSERVATION_NORMALIZATION["speed"],
                        A.car_get_acceleration(agent) / self.OBSERVATION_NORMALIZATION["acceleration"],
                        A.car_get_request_indicated_lane(agent),
                        A.car_get_request_indicated_turn(agent),
                        A.car_get_fuel_level(agent) / self.OBSERVATION_NORMALIZATION["fuel"],
                    ],
                    dtype=np.float32,
                ),
                lane_indicator,
                turn_indicator,
            )
        )

    def _construct_location_observation(self) -> np.ndarray:
        """
        Construct observation for the vehicle's location and lane attributes.

        Returns:
            np.ndarray: Concatenated array of intersection/road position, lane properties,
            and directional information.
        """
        map = A.sim_get_map(self.sim)
        sit = A.sim_get_situational_awareness(self.sim, 0)
        is_on_intersection = sit.is_on_intersection
        center = sit.intersection.center if is_on_intersection else sit.road.mid_point
        lane = sit.lane

        # Determine lane direction for straight or turn lanes
        if lane.type == A.LINEAR_LANE:
            direction_from = direction_to = direction_onehot(lane.direction)
        else:
            lane_prev = A.lane_get_connection_incoming_straight(lane, map)
            lane_next = A.lane_get_connection_straight(lane, map)
            direction_from = direction_onehot(lane_prev.direction)
            direction_to = direction_onehot(lane_next.direction)

        is_straight = lane.type == A.LINEAR_LANE
        is_left_turn = lane.direction == A.DIRECTION_CCW
        is_right_turn = lane.direction == A.DIRECTION_CW

        # Determine lane number and count based on intersection or road
        if is_on_intersection:
            if lane.type == A.LINEAR_LANE:
                prev_lane = A.lane_get_connection_incoming_straight(lane, map)
                prev_road = A.lane_get_road(prev_lane, map)
                num_lanes = prev_road.num_lanes
                lane_number = A.road_find_index_of_lane(prev_road, prev_lane.id)
            else:
                num_lanes = lane_number = 1
        else:
            num_lanes = sit.road.num_lanes
            lane_number = A.road_find_index_of_lane(sit.road, lane.id)

        return np.concatenate(
            (
                np.array(
                    [
                        is_on_intersection,
                        center.x / self.OBSERVATION_NORMALIZATION["center"],
                        center.y / self.OBSERVATION_NORMALIZATION["center"],
                        is_straight,
                        is_left_turn,
                        is_right_turn,
                        sit.is_exit_available,
                        sit.is_exit_available_eventually,
                        sit.is_on_entry_ramp,
                        sit.is_approaching_dead_end,
                        sit.is_an_intersection_upcoming,
                        num_lanes / self.OBSERVATION_NORMALIZATION["lanes"],
                        lane_number / num_lanes,
                        (num_lanes - 1 - lane_number) / num_lanes,
                        A.car_get_lane_progress(self.agent),
                        sit.distance_to_end_of_lane_from_leading_edge / self.OBSERVATION_NORMALIZATION["distance"],
                    ],
                    dtype=np.float32,
                ),
                direction_from,
                direction_to,
            )
        )

    def _construct_driving_assistant_observation(self) -> np.ndarray:
        """
        Construct observation for the ADAS settings.

        Returns:
            np.ndarray: Concatenated array of ADAS parameters and one-hot encoded intents.
        """
        das = A.sim_get_driving_assistant(self.sim, 0)
        sit = A.sim_get_situational_awareness(self.sim, 0)
        stopping_target_from_end_of_lane = sit.distance_to_end_of_lane_from_leading_edge - das.stopping_distance_target if das.stop_mode else 0
        stopping_target_from_end_of_lane = max(stopping_target_from_end_of_lane, 0.0)  # just in case, as it can be negative if we are already past the end of lane and the margin is measured from leading edge
        return np.concatenate(
            (
                np.array(
                    [
                        das.speed_target / self.OBSERVATION_NORMALIZATION["speed"],
                        das.stop_mode,
                        stopping_target_from_end_of_lane / self.OBSERVATION_NORMALIZATION["distance"],
                        das.follow_mode,
                        das.follow_mode_thw / self.OBSERVATION_NORMALIZATION["speed"],
                        das.follow_mode_buffer,
                        das.merge_assistance,
                        das.aeb_assistance,
                        das.aeb_in_progress,
                        das.aeb_manually_disengaged,
                        das.use_preferred_accel_profile,
                        das.use_linear_speed_control,
                    ],
                    dtype=np.float32,
                ),
                indicator_onehot(das.merge_intent),
                indicator_onehot(das.turn_intent),
            )
        )

    def _construct_nearby_vehicles_observation(self) -> np.ndarray:
        """
        Construct observation for nearby vehicles' states.

        Returns:
            np.ndarray: Concatenated array of lead/following vehicle states and nearby vehicle metrics.
        """
        situation = A.sim_get_situational_awareness(self.sim, 0)
        agent_speed = A.car_get_speed(self.agent)
        agent_acc = A.car_get_acceleration(self.agent)

        # Lead vehicle observation
        lead_exists = situation.is_vehicle_ahead
        lead_distance = (
            situation.distance_to_lead_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.lead_vehicle) / 2)
            if lead_exists
            else self.MAX_DISTANCE
        )
        lead_rel_speed = A.car_get_speed(situation.lead_vehicle) - agent_speed if lead_exists else 0
        lead_rel_acc = A.car_get_acceleration(situation.lead_vehicle) - agent_acc if lead_exists else 0

        # Following vehicle observation
        behind_exists = situation.is_vehicle_behind
        behind_distance = (
            situation.distance_to_following_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(situation.following_vehicle) / 2)
            if behind_exists
            else self.MAX_DISTANCE
        )
        behind_rel_speed = A.car_get_speed(situation.following_vehicle) - agent_speed if behind_exists else 0
        behind_rel_acc = A.car_get_acceleration(situation.following_vehicle) - agent_acc if behind_exists else 0

        # Nearby vehicles observation
        nearby_flattened = A.NearbyVehiclesFlattened()
        A.nearby_vehicles_flatten(situation.nearby_vehicles, nearby_flattened, True)
        flattened_count = A.nearby_vehicles_flattened_get_count(nearby_flattened)
        nearby_exists_flags = np.array(
            [A.nearby_vehicles_flattened_get_car_id(nearby_flattened, i) != -1 for i in range(flattened_count)],
            dtype=np.float32,
        )
        nearby_distances = np.array(
            [A.nearby_vehicles_flattened_get_distance(nearby_flattened, i) for i in range(flattened_count)],
            dtype=np.float32,
        ).clip(0, self.MAX_DISTANCE)
        nearby_ttcs = np.array(
            [A.nearby_vehicles_flattened_get_time_to_collision(nearby_flattened, i) for i in range(flattened_count)],
            dtype=np.float32,
        ).clip(0, self.MAX_TTC)
        nearby_thws = np.array(
            [A.nearby_vehicles_flattened_get_time_headway(nearby_flattened, i) for i in range(flattened_count)],
            dtype=np.float32,
        ).clip(0, self.MAX_THW)
        nearby_speeds_relative = np.array(
            [
                (A.car_get_speed(A.sim_get_car(self.sim, A.nearby_vehicles_flattened_get_car_id(nearby_flattened, i))) - agent_speed)
                if nearby_exists_flags[i]
                else 0
                for i in range(flattened_count)
            ],
            dtype=np.float32,
        )
        nearby_acc_relative = np.array(
            [
                (A.car_get_acceleration(A.sim_get_car(self.sim, A.nearby_vehicles_flattened_get_car_id(nearby_flattened, i))) - agent_acc)
                if nearby_exists_flags[i]
                else 0
                for i in range(flattened_count)
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            (
                np.array(
                    [
                        lead_exists,
                        lead_distance / self.OBSERVATION_NORMALIZATION["distance"],
                        lead_rel_speed / self.OBSERVATION_NORMALIZATION["speed"],
                        lead_rel_acc / self.OBSERVATION_NORMALIZATION["acceleration"],
                        behind_exists,
                        behind_distance / self.OBSERVATION_NORMALIZATION["distance"],
                        behind_rel_speed / self.OBSERVATION_NORMALIZATION["speed"],
                        behind_rel_acc / self.OBSERVATION_NORMALIZATION["acceleration"],
                    ]
                ),
                nearby_distances / self.OBSERVATION_NORMALIZATION["distance"],
                nearby_ttcs / self.OBSERVATION_NORMALIZATION["speed"],
                nearby_thws / self.OBSERVATION_NORMALIZATION["speed"],
                nearby_speeds_relative / self.OBSERVATION_NORMALIZATION["speed"],
                nearby_acc_relative / self.OBSERVATION_NORMALIZATION["acceleration"],
            )
        )

    def _construct_environment_conditions_observation(self) -> np.ndarray:
        """
        Construct observation for environmental conditions.

        Returns:
            np.ndarray: Array containing the current lane's speed limit.
        """
        sit = A.sim_get_situational_awareness(self.sim, 0)
        t = A.sim_get_time(self.sim)
        return np.array([sit.lane.speed_limit / self.OBSERVATION_NORMALIZATION["speed"], t / self.OBSERVATION_NORMALIZATION["time"]], dtype=np.float32)

    def _construct_turn_relevant_observations(self) -> np.ndarray:
        """
        Construct observation for turn-related traffic conditions.

        Returns:
            np.ndarray: Array of turn availability and traffic light states.
        """
        sit = A.sim_get_situational_awareness(self.sim, 0)
        lane = sit.lane
        map = A.sim_get_map(self.sim)
        left_turn_available = A.lane_get_connection_left_id(lane) != A.ID_NULL
        straight_available = A.lane_get_connection_straight_id(lane) != A.ID_NULL
        right_turn_available = A.lane_get_connection_right_id(lane) != A.ID_NULL

        is_green = is_yellow = is_red = all_red = is_fourway_stop = False
        countdown = 0
        if intersection := sit.intersection:
            is_fourway_stop = intersection.state == A.FOUR_WAY_STOP
            if not is_fourway_stop:
                direction = (
                    A.lane_get_connection_incoming_straight(lane, map).direction
                    if sit.is_on_intersection
                    else lane.direction
                )
                is_ew = direction in (A.DIRECTION_EAST, A.DIRECTION_WEST)
                is_ns = not is_ew
                is_green = (is_ew and intersection.state == A.NS_RED_EW_GREEN) or (
                    is_ns and intersection.state == A.NS_GREEN_EW_RED
                )
                is_yellow = (is_ew and intersection.state == A.EW_YELLOW_NS_RED) or (
                    is_ns and intersection.state == A.NS_YELLOW_EW_RED
                )
                is_red = not (is_green or is_yellow)
                all_red = intersection.state in (A.ALL_RED_BEFORE_EW_GREEN, A.ALL_RED_BEFORE_NS_GREEN)
                countdown = intersection.countdown
            else:
                all_red = is_red = True

        return np.array(
            [
                left_turn_available,
                straight_available,
                right_turn_available,
                is_green,
                is_yellow,
                is_red,
                all_red,
                is_fourway_stop,
                countdown / self.OBSERVATION_NORMALIZATION["countdown"],
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Build the complete observation vector from all components.

        Returns:
            np.ndarray: Concatenated observation of car state, ADAS, location, nearby vehicles,
            turn conditions, and environment.
        """
        A.situational_awareness_build(self.sim, self.agent.id)
        return np.concatenate(
            (
                self._construct_car_state_observation(),
                self._construct_driving_assistant_observation(),
                self._construct_location_observation(),
                self._construct_nearby_vehicles_observation(),
                self._construct_turn_relevant_observations(),
                self._construct_environment_conditions_observation(),
            )
        )

    def _get_reward(self) -> float:
        """
        Calculate reward based on vehicle speed (used when no goal lane is specified).

        Returns:
            float: Reward proportional to speed in mph, scaled by 1/100.
        """
        return A.to_mph(A.car_get_speed(self.agent)) / 100

    def _crashed(self) -> bool:
        """
        Check if the agent has collided with lead or following vehicle.

        Returns:
            bool: True if a collision occurred, False otherwise.
        """
        situation = A.sim_get_situational_awareness(self.sim, 0)
        for vehicle, distance in [
            (situation.lead_vehicle, situation.distance_to_lead_vehicle),
            (situation.following_vehicle, situation.distance_to_following_vehicle),
        ]:
            if vehicle and distance - (A.car_get_length(self.agent) / 2 + A.car_get_length(vehicle) / 2) <= 0:
                return True
        return False

    def _reached_destination(self) -> bool:
        """
        Check if the agent has reached the goal lane with sufficient progress.

        Returns:
            bool: True if the agent is in the goal lane and has progressed at least 10%, False otherwise.
        """
        if self.goal_lane is None:
            return False
        return A.car_get_lane_id(self.agent) == self.goal_lane and A.car_get_lane_progress(self.agent) > 0.1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Ensures the agent does not start in the goal lane (if specified). Initializes the simulation,
        connects to the rendering server if needed, and configures ADAS features (AEB, follow mode,
        merge assistance).

        Args:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[Dict[str, Any]]): Additional options (unused).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and empty info dictionary.
        """
        super().reset(seed=seed, options=options)
        A.sim_disconnect_from_render_server(self.sim)

        # Reset until agent is not in goal lane (if goal is specified) or a dead-end lane or an the interstate highway (or its entry or exit ramps)
        first_reset_done = False
        while not first_reset_done or (self.goal_lane is not None and A.car_get_lane_id(A.sim_get_car(self.sim, 0)) == self.goal_lane) or A.lane_get_num_connections(A.map_get_lane(A.sim_get_map(self.sim), A.car_get_lane_id(A.sim_get_car(self.sim, 0)))) == 0 or A.lane_get_num_connections_incoming(A.map_get_lane(A.sim_get_map(self.sim), A.car_get_lane_id(A.sim_get_car(self.sim, 0)))) == 0 or A.lane_get_road(A.map_get_lane(A.sim_get_map(self.sim), A.car_get_lane_id(A.sim_get_car(self.sim, 0))), A.sim_get_map(self.sim)).num_lanes >= 3:
            A.awesim_setup(
                self.sim, self.city_width, self.num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY
            )
            first_reset_done = True

        # Configure simulation settings
        if self.synchronized:
            A.sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            A.sim_connect_to_render_server(self.sim, self.render_server_ip, 4242)

        A.sim_set_agent_enabled(self.sim, True)
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent = A.sim_get_agent_car(self.sim)
        self.das = A.sim_get_driving_assistant(self.sim, self.agent.id)
        A.driving_assistant_reset_settings(self.das, self.agent)

        # Configure ADAS features
        A.driving_assistant_configure_aeb_assistance(self.das, self.agent, self.sim, True)
        A.driving_assistant_configure_merge_assistance(self.das, self.agent, self.sim, True)
        A.driving_assistant_configure_follow_mode(self.das, self.agent, self.sim, False)
        A.driving_assistant_configure_stop_mode(self.das, self.agent, self.sim, False)

        # Configure fuel
        A.car_set_fuel_level(self.agent, A.from_gallons(self.init_fuel_gallons))
        A.car_set_fuel_tank_capacity(self.agent, A.from_gallons(self.init_fuel_gallons))    # unnecessary but just in case as default tank capacity is Inf for all cars in awesim

        self.steps = 0
        self.fuel_drive_beginning = A.car_get_fuel_level(self.agent)

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment based on the provided action.

        The action controls:
        1. Speed adjustment (±10 mph).
        2. Turn or merge intent (left/right/none, with contextual validation).
        3. Follow time-headway adjustment (±1 second).

        The simulation advances in sub-steps to check for crashes, goal achievement, or timeout.
        Rewards include time penalty, crash penalty, goal reward, and speed-based reward (if no goal).

        Args:
            action (np.ndarray): 3D array for speed, turn/merge, and follow time-headway adjustments.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward, terminated flag,
            truncated flag, and info dictionary (time, success, crash status).
        """
        A.situational_awareness_build(self.sim, self.agent.id)
        sit = A.sim_get_situational_awareness(self.sim, self.agent.id)

        action_follow_mode_probability = action[0]
        action_speed_target = action[1]
        action_follow_mode_thw = action[2]
        action_margin = action[3]
        action_turn_signal_probability = action[4]


        # clear ADAS mode first
        A.driving_assistant_configure_follow_mode(self.das, self.agent, self.sim, False)
        A.driving_assistant_configure_stop_mode(self.das, self.agent, self.sim, False)

        # decide mode.
        follow_mode = self.np_random.random() < action_follow_mode_probability
        stop_mode = not follow_mode
        if (stop_mode):
            # if there is no intersection or dead end upcoming, then stay in follow mode
            if not (sit.is_an_intersection_upcoming or sit.is_on_intersection or sit.is_approaching_dead_end or sit.is_on_entry_ramp):
                follow_mode = True
                stop_mode = False
            # if there is a lead vehicle, then stay in follow mode
            if sit.is_vehicle_ahead:
                follow_mode = True
                stop_mode = False
            # cannot do stop mode if we are reversing
            if A.car_get_speed(self.agent) < 0:
                follow_mode = True
                stop_mode = False

        # if (sit.is_an_intersection_upcoming or sit.is_approaching_dead_end or sit.is_on_entry_ramp) and A.car_get_speed(self.agent) > 0 and not sit.is_vehicle_ahead:
        #     stop_mode = True
        #     follow_mode = not stop_mode

        # set the mode settings, then enable the mode in ADAS

        if follow_mode:
            # Speed adjustment
            speed_target = float(action_speed_target) * self.agent.capabilities.top_speed
            A.driving_assistant_configure_speed_target(self.das, self.agent, self.sim, A.mps(speed_target))

            # Follow mode time-headway adjustment
            new_follow_thw = float(action_follow_mode_thw)
            A.driving_assistant_configure_follow_mode_thw(self.das, self.agent, self.sim, A.seconds(new_follow_thw))

            # Follow mode margin:
            new_margin = float(action_margin + 1e-8)
            A.driving_assistant_configure_follow_mode_buffer(self.das, self.agent, self.sim, new_margin)
            # Set stopping distance target to 0 since it is unused in follow mode
            A.driving_assistant_configure_stopping_distance_target(self.das, self.agent, self.sim, 0.0)

            # set follow mode
            A.driving_assistant_configure_follow_mode(self.das, self.agent, self.sim, True)

        else:   # stop mode
            # speed adjustment. Has to be 0.
            speed_target = 0.0
            A.driving_assistant_configure_speed_target(self.das, self.agent, self.sim, A.mps(speed_target))

            # follow mode time-headway adjustment. Has no effect in stop mode, but still, since it is used for deciding merge safety, it should be set.
            new_follow_thw = float(action_follow_mode_thw)
            A.driving_assistant_configure_follow_mode_thw(self.das, self.agent, self.sim, A.seconds(new_follow_thw))

            # decide stopping distance based on margin from end of lane.
            new_margin = float(action_margin)
            stopping_distance = sit.distance_to_end_of_lane_from_leading_edge - new_margin
            # if we are already past the stopping point, then set stopping distance to 0 so that we stop immediately
            if stopping_distance < 0:
                stopping_distance = 0.0
                new_margin = sit.distance_to_end_of_lane_from_leading_edge - stopping_distance
                new_margin = max(new_margin, 0.0)  # just in case. Happens when we are already past the end of lane and the margin is negative because we measured from leading edge
            A.driving_assistant_configure_stopping_distance_target(self.das, self.agent, self.sim, stopping_distance)
            # set follow mode buffer to a default value, since it is still used for deciding merge safety
            A.driving_assistant_configure_follow_mode_buffer(self.das, self.agent, self.sim, A.from_feet(3))

            # set stop mode
            A.driving_assistant_configure_stop_mode(self.das, self.agent, self.sim, True)

        # Merge/turn intent
        should_indicate = self.np_random.random() < abs(action_turn_signal_probability)
        indicator_direction = (
            (A.INDICATOR_LEFT if action_turn_signal_probability < 0 else A.INDICATOR_RIGHT) if should_indicate else A.INDICATOR_NONE
        )
        new_merge_intent = new_turn_intent = A.INDICATOR_NONE
        if indicator_direction != A.INDICATOR_NONE:
            turn_makes_sense = (
                indicator_direction == A.INDICATOR_LEFT and A.lane_get_connection_left_id(sit.lane) != A.ID_NULL
            ) or (indicator_direction == A.INDICATOR_RIGHT and A.lane_get_connection_right_id(sit.lane) != A.ID_NULL)
            merge_makes_sense = (
                indicator_direction == A.INDICATOR_LEFT and sit.is_lane_change_left_possible
            ) or (
                indicator_direction == A.INDICATOR_RIGHT and (sit.is_lane_change_right_possible or sit.is_exit_available_eventually)
            )
            if turn_makes_sense and not merge_makes_sense:
                new_turn_intent = indicator_direction
            elif merge_makes_sense and not turn_makes_sense:
                new_merge_intent = indicator_direction
                if (
                    sit.is_an_intersection_upcoming and sit.distance_to_end_of_lane_from_leading_edge < A.meters(10) and A.car_get_speed(self.agent) > A.from_mph(5)
                ):
                    new_merge_intent = A.INDICATOR_NONE  # Prevent last-moment lane changes near intersections
            elif not (turn_makes_sense or merge_makes_sense):
                new_merge_intent = A.INDICATOR_NONE
                new_turn_intent = A.INDICATOR_NONE
            elif sit.is_approaching_end_of_lane:
                new_turn_intent = indicator_direction
            else:
                new_merge_intent = indicator_direction

        A.driving_assistant_configure_merge_intent(self.das, self.agent, self.sim, new_merge_intent)
        A.driving_assistant_configure_turn_intent(self.das, self.agent, self.sim, new_turn_intent)

        if self.verbose:
            print(
                f"Mode: {'Follow' if follow_mode else 'Stop'}, "
                f"Action: Speed cap: {A.to_mph(speed_target):.1f} mph (cur = {A.to_mph(A.car_get_speed(self.agent)):.1f} mph), "
                f"Follow THW: {new_follow_thw:.2f} sec, "
                f"Margin buffer: {A.to_feet(new_margin):.1f} ft (cur = {A.to_feet(sit.distance_to_end_of_lane_from_leading_edge if stop_mode else np.nan):.1f} ft), "
                f"{'Merge' if new_merge_intent != A.INDICATOR_NONE else 'Turn' if new_turn_intent != A.INDICATOR_NONE else 'No'} "
                f"{'Left' if new_merge_intent == A.INDICATOR_LEFT or new_turn_intent == A.INDICATOR_LEFT else 'Right' if new_merge_intent == A.INDICATOR_RIGHT or new_turn_intent == A.INDICATOR_RIGHT else ''}, "
            )

        # Simulate and compute reward
        reward = 0.0
        cost = 0.0
        elapsed_time = 0.0
        sim_time_initial = A.sim_get_time(self.sim)
        crashed = reached_goal = timeout = out_of_fuel = False
        crashed_forward = crashed_backward = False
        crashed_on_intersection = False
        fuel_initial = A.car_get_fuel_level(self.agent)
        while not (crashed or reached_goal or timeout or out_of_fuel) and elapsed_time < self.decision_interval - 1e-6:
            delta_time = min(self.TERMINATION_CHECK_INTERVAL, self.decision_interval - elapsed_time)    # make sure you don't end up simulating more than decision_interval if it is not a multiple of TERMINATION_CHECK_INTERVAL
            t0 = A.sim_get_time(self.sim)
            A.simulate(self.sim, delta_time)  # Advance simulation
            delta_time = A.sim_get_time(self.sim) - t0  # true delta time (since it is not guaranteed that sim would have simulated exactly for delta_time if it is not a multiple of dt.)
            elapsed_time = A.sim_get_time(self.sim) - sim_time_initial
            A.situational_awareness_build(self.sim, self.agent.id)
            reward -= self.time_penalty_per_second * delta_time  # Apply time penalty
            if self.goal_lane is None:
                reward += A.car_get_speed(self.agent) * delta_time  # Reward for speed
            crashed = self._crashed()
            reached_goal = self._reached_destination()
            timeout = A.sim_get_time(self.sim) >= self.sim_duration
            out_of_fuel = A.car_get_fuel_level(self.agent) < 1e-9
            if crashed:
                reward -= self.crash_penalty
                cost += self.cost_car
                crashed_forward = sit.is_vehicle_ahead and sit.distance_to_lead_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(sit.lead_vehicle) / 2) <= 0
                crashed_backward = sit.is_vehicle_behind and sit.distance_to_following_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(sit.following_vehicle) / 2) <= 0
                if sit.is_on_intersection:
                    crashed_on_intersection = True
            if reached_goal:
                reward += self.goal_reward

        fuel_final = A.car_get_fuel_level(self.agent)
        fuel_consumed = fuel_initial - fuel_final
        fuel_consumed_gallons = A.to_gallons(fuel_consumed)
        if np.isfinite(fuel_consumed_gallons):
            cost += self.cost_gas_per_gallon * fuel_consumed_gallons

        is_late = A.sim_get_time(self.sim) > self.appointment_time

        # Start accumulating lost wage cost if appointment time has passed and goal lane is not reached yet
        if self.goal_lane is not None and not reached_goal and is_late:
            cost += self.cost_lost_wage_per_hour * self.decision_interval / 3600.0

        if self.may_lose_entire_day_wage and self.goal_lane is not None and not reached_goal and (timeout or out_of_fuel or crashed):
            how_late_already_hrs = (A.sim_get_time(self.sim) - self.appointment_time) / 3600.0
            how_late_already_hrs = max(how_late_already_hrs, 0)  # in case it is not time for appointment yet
            cost_lost_wage_already_considered = how_late_already_hrs * self.cost_lost_wage_per_hour
            workday_wage = 8 * self.cost_lost_wage_per_hour
            cost += max(workday_wage - cost_lost_wage_already_considered, 0)  # Cap lost wage cost to a full workday if never reached the goal lane

        if self.verbose:
            if crashed:
                print("Crashed!")
            if reached_goal:
                if is_late:
                    print("Reached goal, but late!")
                else:
                    print("Reached goal!")
            if out_of_fuel:
                print("Out of fuel!")
            if timeout:
                print("Timed out!")

        obs = self._get_observation()
        info = {
            "time": A.sim_get_time(self.sim),
            "is_success": not crashed if self.goal_lane is None else reached_goal,
            "crashed": crashed,
            "crashed_on_intersection": crashed_on_intersection,
            "crashed_forward": crashed_forward,
            "crashed_backward": crashed_backward,
            "progress_m_during_crash": A.car_get_lane_progress_meters(self.agent) if crashed else np.nan,
            "progress_m_during_front_crash": A.car_get_lane_progress_meters(self.agent) if crashed_forward else np.nan,
            "progress_m_during_back_crash": A.car_get_lane_progress_meters(self.agent) if crashed_backward else np.nan,
            "aeb_in_progress_during_crash": float(self.das.aeb_in_progress) if crashed else np.nan,
            "reached_goal": reached_goal,
            "reached_in_time": reached_goal and not is_late,
            "reached_but_late": reached_goal and is_late,
            "out_of_fuel": out_of_fuel,
            "total_fuel_consumed": A.to_gallons(self.fuel_drive_beginning - fuel_final),
            "timeout": timeout,
            "cost": cost,
        }
        self.steps += 1

        terminated = crashed or reached_goal or out_of_fuel or timeout
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Clean up simulation resources and disconnect from the rendering server.
        """
        A.sim_disconnect_from_render_server(self.sim)
        A.sim_free(self.sim)
        super().close()
