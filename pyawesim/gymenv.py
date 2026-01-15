from typing import Any, Optional, Tuple, Dict, Union
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bindings as A


class AwesimEnv(gym.Env):
    # Constants
    DEFAULT_CITY_WIDTH = 1000  # City width in meters
    DEFAULT_NUM_CARS = 256  # Number of vehicles in the simulation
    DEFAULT_DECISION_INTERVAL = 1.0  # Time between agent decisions (seconds)
    DEFAULT_SIM_DURATION = 60 * 60  # Simulation duration (seconds, default 1 hour)
    DEFAULT_APPOINTMENT_TIME = 60 * 30  # Default appointment time (seconds, 30 minutes)
    DEFAULT_TIME_PENALTY = 0.0  # Penalty per second for time spent
    DEFAULT_GOAL_REWARD = 1.0  # Reward for reaching the goal lane
    DEFAULT_CRASH_PENALTY = 0.0  # Penalty for crashing
    DEFAULT_RENDER_IP = "127.0.0.1"  # Default IP for rendering server
    TERMINATION_CHECK_INTERVAL = 0.1  # Interval for checking termination conditions (seconds). Also, the frequency with which camera frames are captured
    MIN_THW = 0.2  # Minimum follow time-headway (seconds)
    MAX_THW = 5.0  # Maximum follow time-headway (seconds)
    COST_GAS_PER_GALLON = 3.2  # Average cost of gas per gallon ($) in US
    COST_LOST_WAGE_PER_HOUR = 30.0  # Average wages per hour ($) in US
    MAY_LOSE_ENTIRE_DAY_WAGE = False  # Lose entire 8-hour workday wage if not reached by the end of sim_duration
    # COST_CAR = 35000.0  # Average new car cost ($) in US. This is the cost for total loss in a crash.
    COST_CAR = 0.0  # Set to 0 for now to not penalize crashes too much
    DEFAULT_INIT_FUEL_GALLONS = 15.0  # Default initial fuel level in gallons
    DEFAULT_CAM_RESOLUTION = (128, 128)  # Default camera resolution (width, height)
    OBSERVATION_NORMALIZATION = {
        "speed": A.from_mph(100),  # Normalize speeds by 100 mph
        "acceleration": 12,  # Normalize accelerations by 12 m/s²
        "distance": 100,  # Normalize distances by 100 meters
        "center": 1000,  # Normalize coordinates by 1000 meters
        "lanes": 10,  # Normalize lane counts by 10
        "countdown": 60,  # Normalize traffic light countdown by 60 seconds
        'time': 3600,  # Normalize time by 3600 seconds (1 hour)
        'fuel': 50,  # Normalize fuel level by 50 liters (~13.2 gallons)
    }

    def __init__(
        self,
        use_das: bool = True,
        merge_assist: bool = True,
        follow_assist: bool = True,
        aeb_assist: bool = True,
        stop_assist: bool = True,
        deprecated_info_display: bool = False,
        cam_resolution: Tuple[int, int] = DEFAULT_CAM_RESOLUTION,
        anti_aliasing: bool = False,
        reward_shaping: bool = False,
        reward_shaping_gamma: float = 0.99,
        framestack: int = 4,
        city_width: int = DEFAULT_CITY_WIDTH,
        goal_lane: Optional[Union[int, str]] = "random",
        num_cars: int = DEFAULT_NUM_CARS,
        decision_interval: float = DEFAULT_DECISION_INTERVAL,
        sim_duration: float = DEFAULT_SIM_DURATION,
        appointment_time: float = DEFAULT_APPOINTMENT_TIME,  # time in seconds by which the agent should reach the goal lane, otherwise lost wage cost will be start accumulating
        time_penalty_per_second: float = DEFAULT_TIME_PENALTY,
        goal_reward: float = DEFAULT_GOAL_REWARD,
        crash_penalty: float = DEFAULT_CRASH_PENALTY,
        crash_ends_episode: bool = True,
        cost_gas_per_gallon: float = COST_GAS_PER_GALLON,
        cost_lost_wage_per_hour: float = COST_LOST_WAGE_PER_HOUR,
        may_lose_entire_day_wage: bool = MAY_LOSE_ENTIRE_DAY_WAGE,
        cost_car: float = COST_CAR,
        init_fuel_gallons: float = DEFAULT_INIT_FUEL_GALLONS,
        deterministic_actions: bool = False,
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
            cam_resolution (Tuple[int, int]): Resolution of the agent's camera (width, height).
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
            deterministic_actions (bool): If True, actions to select follow-mode vs stop-mode and turn/merge signals are deterministic instead of bernoulli-sampled.
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
        self.use_das = use_das
        self.merge_assist = merge_assist
        self.follow_assist = follow_assist
        self.aeb_assist = aeb_assist
        self.stop_assist = stop_assist
        self.deprecated_info_display = deprecated_info_display
        self.cam_resolution = cam_resolution
        self.anti_aliasing = anti_aliasing
        self.reward_shaping = reward_shaping
        self.reward_shaping_gamma = reward_shaping_gamma
        self.framestack = framestack
        self.goal_lane = goal_lane
        self.goal_select_random = goal_lane == "random"
        self.city_width = city_width
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.time_penalty_per_second = time_penalty_per_second
        self.goal_reward = goal_reward
        self.crash_penalty = crash_penalty
        self.crash_ends_episode = crash_ends_episode
        self.cost_gas_per_gallon = cost_gas_per_gallon
        self.cost_lost_wage_per_hour = cost_lost_wage_per_hour
        self.may_lose_entire_day_wage = may_lose_entire_day_wage
        self.cost_car = cost_car
        self.init_fuel_gallons = init_fuel_gallons
        self.deterministic_actions = deterministic_actions
        self.sim_duration = sim_duration
        self.appointment_time = appointment_time
        self.steps = 0
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render
        self.render_server_ip = render_server_ip
        self.verbose = verbose
        self.render_mode = "rgb_array"

        # Initialize simulation
        self.sim = A.sim_malloc()  # Allocate simulation instance
        A.awesim_setup(
            self.sim, self.city_width, self.num_cars, 0.02, A.Monday_8_AM, A.WEATHER_SUNNY
        )  # Set up city with fixed time (8 AM) and sunny weather
        A.sim_set_agent_enabled(self.sim, True)  # Enable agent control
        if self.use_das:
            A.sim_set_agent_driving_assistant_enabled(self.sim, True)  # Enable ADAS
        self.agent = A.sim_get_agent_car(self.sim)  # Get agent vehicle
        if self.verbose:
            print(f"Agent car ID: {self.agent.id}")
        self.cams = [A.rgbcam_malloc(A.coordinates_create(0, 0), 0, 0, cam_resolution[0], cam_resolution[1], A.from_degrees(0), A.meters(0)) for _ in range(7)]  # Initialize 7 cameras
        if self.anti_aliasing:
            for cam in self.cams:
                A.rgbcam_set_aa_level(cam, 2)  # 2x anti-aliasing
        self.infos_display = A.infos_display_malloc(cam_resolution[0], cam_resolution[1], len(self._get_info_for_info_display()))
        self.minimap = A.minimap_malloc(cam_resolution[0], cam_resolution[1], A.sim_get_map(self.sim))
        self.collision_checker = A.collisions_malloc()
        self.path_planner = A.path_planner_create(A.sim_get_map(self.sim), False)
        if self.goal_select_random:
            self.sample_random_goal()
        self.goal_lane_midpoint = A.map_get_lane(A.sim_get_map(self.sim), self.goal_lane).mid_point if self.goal_lane is not None else None
        self.prev_opt_distance_to_goal = None

        print("seeding rng with ", env_index + 1)
        A.seed_rng(env_index + 1)  # Seed random number generator for reproducibility

        if self.use_das:
            # action space: speed, thw, indicator, stop_at_intersection
            speed_target_low, speed_target_high = A.from_mph(0), A.from_mph(100)  # in m/s
            thw_low, thw_high = self.MIN_THW, self.MAX_THW  # in seconds
            turn_signal_min, turn_signal_max = -1.0, 1.0                                # -1 means left signal, +1 means right signal, 0 means no signal, anything in between means probabilistically choose left / right with probability proportional to magnitude of the value
            stop_prob_min, stop_prob_max = 0.0, 1.0  # 0 means no stop, 1 means stop at intersection. anything in between means probabilistically choose stop / no stop with probability proportional to magnitude of the value

            lows, highs = [], []
            lows.append(speed_target_low)
            highs.append(speed_target_high)
            if self.follow_assist:
                lows.append(thw_low)
                highs.append(thw_high)
            lows.append(turn_signal_min)
            highs.append(turn_signal_max)
            if self.stop_assist:
                lows.append(stop_prob_min)
                highs.append(stop_prob_max)
            self.action_space = spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
            )
        else:
            # action space: 2 components: acceleration [-1, 1] scaled to capabilities, turn / merge intent [-1, 1] with probability proportional to magnitude. There is not merge assist or follow assist.
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )

        # Define observation space. 9 images: 7 cameras (front main, front wide, front-left, front-right, rear, rear-left, rear-right), one minimap and one image displaying various vehicle, adas and environment state variables.
        # The images are stacked along the channel dimension for each camera, and then stacked along a new axis for each camera.
        # Final shape: (9, 3 * framestack, height, width)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(9, 3 * framestack, cam_resolution[1], cam_resolution[0]), dtype=np.uint8
        )

    def sample_random_goal(self):
        map = A.sim_get_map(self.sim)
        while True:
            random_road = A.map_get_road(map, A.rand_int_range(0, map.num_roads - 1))
            random_lane = A.road_get_lane(random_road, map, A.rand_int_range(0, A.road_get_num_lanes(random_road) - 1))
            if A.lane_get_length(random_lane) > 50:
                break
        self.goal_lane = random_lane.id
        self.goal_lane_midpoint = A.map_get_lane(A.sim_get_map(self.sim), self.goal_lane).mid_point if self.goal_lane is not None else None

    def _get_info_for_info_display(self) -> list[Tuple[float, A.RGB]]:

        if self.use_das:
            A.situational_awareness_build(self.sim, self.agent.id)
            das = A.sim_get_driving_assistant(self.sim, self.agent.id)
            sit = A.sim_get_situational_awareness(self.sim, self.agent.id)

            # car state
            fuel = A.car_get_fuel_level(self.agent) / self.OBSERVATION_NORMALIZATION["fuel"]
            speed = A.car_get_speed(self.agent) / self.OBSERVATION_NORMALIZATION["speed"]
            accel = A.car_get_acceleration(self.agent) / self.OBSERVATION_NORMALIZATION["acceleration"]
            ind_turn_left = A.car_get_indicator_turn(self.agent) == A.INDICATOR_LEFT
            ind_merge_left = A.car_get_indicator_lane(self.agent) == A.INDICATOR_LEFT
            ind_left = ind_turn_left or ind_merge_left
            ind_turn_right = A.car_get_indicator_turn(self.agent) == A.INDICATOR_RIGHT
            ind_merge_right = A.car_get_indicator_lane(self.agent) == A.INDICATOR_RIGHT
            ind_right = ind_turn_right or ind_merge_right

            # ADAS state
            speed_target = das.speed_target / self.OBSERVATION_NORMALIZATION["speed"]
            thw_target = das.thw / self.MAX_THW
            should_stop_at_intersection = das.should_stop_at_intersection
            aeb_in_progress = das.aeb_in_progress

            # environment state
            t = A.sim_get_time(self.sim) / self.OBSERVATION_NORMALIZATION["time"]
            speed_limit = sit.lane.speed_limit / self.OBSERVATION_NORMALIZATION["speed"]
            my_turn_fcfs = float(sit.is_my_turn_at_stop_sign_fcfs)

            infos_and_colors = [
                (fuel, A.RGB_WHITE),
                (abs(speed), A.RGB_BLUE if speed >= 0 else A.RGB_YELLOW),
                (abs(accel), A.RGB_GREEN if accel >= 0 else A.RGB_RED),

                (float(ind_left), A.RGB_ORANGE),
                (abs(speed_target), A.RGB_MAGENTA if speed_target >= 0 else A.RGB_YELLOW),
                (float(ind_right), A.RGB_ORANGE),

                (float(should_stop_at_intersection), A.RGB_RED),
                (thw_target, A.RGB_CYAN),
                (float(aeb_in_progress), A.RGB_RED),

                (speed_limit, A.RGB_WHITE),
                (my_turn_fcfs, A.RGB_GREEN),
                (t, A.RGB_WHITE),
            ]
        else:
            # car state
            fuel = A.car_get_fuel_level(self.agent) / self.OBSERVATION_NORMALIZATION["fuel"]
            speed = A.car_get_speed(self.agent) / self.OBSERVATION_NORMALIZATION["speed"]

            # environment state
            t = A.sim_get_time(self.sim) / self.OBSERVATION_NORMALIZATION["time"]

            infos_and_colors = [
                (fuel, A.RGB_RED if fuel < A.from_gallons(1) else A.RGB_GREEN),
                (abs(speed), A.RGB_BLUE if speed >= 0 else A.RGB_YELLOW),
                (t, A.RGB_WHITE),
            ]

        return infos_and_colors

    def _get_info_for_info_display_deprecated(self) -> list[Tuple[float, A.RGB]]:

        if self.use_das:
            A.situational_awareness_build(self.sim, self.agent.id)
            das = A.sim_get_driving_assistant(self.sim, self.agent.id)
            sit = A.sim_get_situational_awareness(self.sim, self.agent.id)

            # car state
            fuel = A.car_get_fuel_level(self.agent) / self.OBSERVATION_NORMALIZATION["fuel"]
            speed = A.car_get_speed(self.agent) / self.OBSERVATION_NORMALIZATION["speed"]
            accel = A.car_get_acceleration(self.agent) / self.OBSERVATION_NORMALIZATION["acceleration"]
            ind_turn_left = A.car_get_indicator_turn(self.agent) == A.INDICATOR_LEFT
            ind_merge_left = A.car_get_indicator_lane(self.agent) == A.INDICATOR_LEFT
            ind_left = ind_turn_left or ind_merge_left
            ind_turn_right = A.car_get_indicator_turn(self.agent) == A.INDICATOR_RIGHT
            ind_merge_right = A.car_get_indicator_lane(self.agent) == A.INDICATOR_RIGHT
            ind_right = ind_turn_right or ind_merge_right

            # ADAS state
            speed_target = das.speed_target / self.OBSERVATION_NORMALIZATION["speed"]
            thw_target = das.thw / self.MAX_THW
            merge_intent_left = das.merge_intent == A.INDICATOR_LEFT
            merge_intent_right = das.merge_intent == A.INDICATOR_RIGHT
            turn_intent_left = das.turn_intent == A.INDICATOR_LEFT
            turn_intent_right = das.turn_intent == A.INDICATOR_RIGHT
            intent_left = merge_intent_left or turn_intent_left
            intent_right = merge_intent_right or turn_intent_right
            should_stop_at_intersection = das.should_stop_at_intersection
            aeb_in_progress = das.aeb_in_progress

            # environment state
            t = A.sim_get_time(self.sim) / self.OBSERVATION_NORMALIZATION["time"]
            speed_limit = sit.lane.speed_limit / self.OBSERVATION_NORMALIZATION["speed"]

            infos_and_colors = [
                (fuel, A.RGB_WHITE), (abs(speed), A.RGB_BLUE if speed >= 0 else A.RGB_YELLOW), (abs(accel), A.RGB_GREEN if accel >= 0 else A.RGB_RED),
                (float(ind_left), A.RGB_RED if ind_turn_left else (A.RGB_ORANGE if ind_merge_left else A.RGB_WHITE)), (float(ind_right), A.RGB_RED if ind_turn_right else (A.RGB_ORANGE if ind_merge_right else A.RGB_WHITE)), (abs(speed_target), A.RGB_MAGENTA if speed_target >= 0 else A.RGB_CYAN),
                (thw_target, A.RGB_MAGENTA), (float(intent_left), A.RGB_RED if turn_intent_left else (A.RGB_ORANGE if merge_intent_left else A.RGB_WHITE)), (float(intent_right), A.RGB_RED if turn_intent_right else (A.RGB_ORANGE if merge_intent_right else A.RGB_WHITE)),
                (float(should_stop_at_intersection), A.RGB_CYAN), (float(aeb_in_progress), A.RGB_RED), (t, A.RGB_WHITE), (speed_limit, A.RGB_WHITE),
            ]
        else:
            # car state
            fuel = A.car_get_fuel_level(self.agent) / self.OBSERVATION_NORMALIZATION["fuel"]
            speed = A.car_get_speed(self.agent) / self.OBSERVATION_NORMALIZATION["speed"]

            # environment state
            t = A.sim_get_time(self.sim) / self.OBSERVATION_NORMALIZATION["time"]

            infos_and_colors = [
                (fuel, A.RGB_RED if fuel < A.from_gallons(1) else A.RGB_GREEN),
                (abs(speed), A.RGB_BLUE if speed >= 0 else A.RGB_YELLOW),
                (t, A.RGB_WHITE),
            ]

        return infos_and_colors

    def _get_obs_as_list_of_images(self) -> list[np.ndarray]:
        # ensure cams are attached to the agent
        A.rgbcam_attach_to_car(self.cams[0], self.agent, A.CAR_CAMERA_MAIN_FORWARD)
        A.rgbcam_attach_to_car(self.cams[1], self.agent, A.CAR_CAMERA_WIDE_FORWARD)
        A.rgbcam_attach_to_car(self.cams[2], self.agent, A.CAR_CAMERA_SIDE_LEFT_FORWARDVIEW)
        A.rgbcam_attach_to_car(self.cams[3], self.agent, A.CAR_CAMERA_SIDE_RIGHT_FORWARDVIEW)
        A.rgbcam_attach_to_car(self.cams[4], self.agent, A.CAR_CAMERA_SIDE_LEFT_REARVIEW)
        A.rgbcam_attach_to_car(self.cams[5], self.agent, A.CAR_CAMERA_SIDE_RIGHT_REARVIEW)
        A.rgbcam_attach_to_car(self.cams[6], self.agent, A.CAR_CAMERA_REARVIEW)

        # capture all cams
        for cam in self.cams:
            A.rgbcam_capture(cam, self.sim)

        # minimap render
        self.minimap.marked_car_id = self.agent.id
        self._get_opt_dist_to_goal()  # update path planner
        self.minimap.marked_path = self.path_planner
        A.minimap_render(self.minimap, self.sim)

        # info display render
        infos_and_colors = self._get_info_for_info_display() if not self.deprecated_info_display else self._get_info_for_info_display_deprecated()
        for i, (info, color) in enumerate(infos_and_colors):
            A.infos_display_set_info(self.infos_display, i, info)
            A.infos_display_set_info_color(self.infos_display, i, color)
        A.infos_display_render(self.infos_display)

        return [
            self.cams[0].to_numpy(),
            self.cams[1].to_numpy(),
            self.cams[2].to_numpy(),
            self.cams[3].to_numpy(),
            self.cams[4].to_numpy(),
            self.cams[5].to_numpy(),
            self.cams[6].to_numpy(),
            self.minimap.to_numpy(),
            self.infos_display.to_numpy(),
        ]

    def _get_opt_dist_to_goal(self) -> float:
        # cur_pos = A.car_get_center(self.agent)
        # goal_pos = self.goal_lane_midpoint
        # if goal_pos is None:
        #     return 0.0
        # return abs(cur_pos.x - goal_pos.x) + abs(cur_pos.y - goal_pos.y)
        self_lane = A.car_get_lane(self.agent, A.sim_get_map(self.sim))
        goal_lane = A.map_get_lane(A.sim_get_map(self.sim), self.goal_lane)
        A.path_planner_compute_shortest_path(self.path_planner, self_lane, A.car_get_lane_progress_meters(self.agent), goal_lane, 0.5 * A.lane_get_length(goal_lane))
        dist = float(self.path_planner.optimal_cost)
        dist = min(dist, self.city_width * 10)  # cap to avoid numerical issues
        return dist

    def _get_observation(self) -> np.ndarray:
        # self.frame_buffer contains `framestack` elements.
        # Each element is a list of 9 arrays of shape (3, H, W).
        # We want (9, 3 * framestack, H, W).

        buffer_list = list(self.frame_buffer)
        num_cams = 9

        final_obs = []
        for cam_idx in range(num_cams):
            # Collect frames for this camera
            cam_frames = [step_obs[cam_idx]
                          for step_obs in buffer_list]  # List of (3, H, W)
            # Concatenate along channel dimension (axis 0)
            stacked_cam = np.concatenate(
                cam_frames, axis=0)  # (3 * framestack, H, W)
            final_obs.append(stacked_cam)

        return np.stack(final_obs, axis=0)  # (9, 3 * framestack, H, W)

    def _get_reward(self) -> float:
        """
        Calculate reward based on vehicle speed (used when no goal lane is specified).

        Returns:
            float: Reward proportional to speed in mph, scaled by 1/100.
        """
        return A.to_mph(A.car_get_speed(self.agent)) / 100

    def _crashed(self) -> bool:
        """
        Check if the agent has collided with any vehicle.

        Returns:
            bool: True if a collision occurred, False otherwise.
        """
        A.collisions_detect_for_car(self.collision_checker, self.sim, self.agent.id)
        return A.collisions_get_count_for_car(self.collision_checker, self.agent.id) > 0

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

        if self.goal_select_random:
            self.sample_random_goal()

        # Reset until agent is not in goal lane (if goal is specified) or a dead-end lane or an the interstate highway (or its entry or exit ramps)
        first_reset_done = False
        num_cars = max(np.random.randint(self.num_cars // 4, self.num_cars + 1), 16)    # randomize number of cars a bit. At least 16 cars.
        while not first_reset_done or (self.goal_lane is not None and A.car_get_lane_id(A.sim_get_car(self.sim, 0)) == self.goal_lane) or A.lane_get_num_connections(A.map_get_lane(A.sim_get_map(self.sim), A.car_get_lane_id(A.sim_get_car(self.sim, 0)))) == 0 or A.lane_get_num_connections_incoming(A.map_get_lane(A.sim_get_map(self.sim), A.car_get_lane_id(A.sim_get_car(self.sim, 0)))) == 0:
            A.awesim_setup(
                self.sim, self.city_width, num_cars, 0.02, A.clock_reading(0, 8, 0, 0), A.WEATHER_SUNNY
            )
            first_reset_done = True
        self.prev_opt_distance_to_goal = self._get_opt_dist_to_goal()

        # Configure simulation settings
        if self.synchronized:
            A.sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            A.sim_connect_to_render_server(self.sim, self.render_server_ip, 4242)

        A.sim_set_agent_enabled(self.sim, True)
        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=self.framestack)

        self.agent = A.sim_get_agent_car(self.sim)
        if self.use_das:
            # Enable ADAS features
            A.sim_set_agent_driving_assistant_enabled(self.sim, True)
            self.das = A.sim_get_driving_assistant(self.sim, self.agent.id)
            A.driving_assistant_reset_settings(self.das, self.agent)
            if not self.merge_assist:
                A.driving_assistant_configure_merge_assistance(self.das, self.agent, self.sim, False)
            if not self.follow_assist:
                A.driving_assistant_configure_follow_assistance(self.das, self.agent, self.sim, False)
            if not self.aeb_assist:
                A.driving_assistant_configure_aeb_assistance(self.das, self.agent, self.sim, False)

        # Configure collision checker
        A.collisions_reset(self.collision_checker)

        # Configure fuel
        A.car_set_fuel_level(self.agent, A.from_gallons(self.init_fuel_gallons))
        A.car_set_fuel_tank_capacity(self.agent, A.from_gallons(self.init_fuel_gallons))    # unnecessary but just in case as default tank capacity is Inf for all cars in awesim

        self.steps = 0
        self.fuel_drive_beginning = A.car_get_fuel_level(self.agent)

        # Capture initial observation
        initial_frames = self._get_obs_as_list_of_images()
        processed_frames = [f.transpose(2, 0, 1).copy()
                            for f in initial_frames]  # (3, H, W)

        # Fill with initial frame
        for _ in range(self.framestack):
            self.frame_buffer.append(processed_frames)

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

        if self.use_das:
            a_id = 0
            action_speed_target = action[a_id]
            a_id += 1
            if self.follow_assist:
                action_thw_target = action[a_id]
                a_id += 1
            else:
                action_thw_target = None
            action_indicator_probability = action[a_id]
            a_id += 1
            if self.stop_assist:
                action_stop_at_intersection_probability = action[a_id]
            else:
                action_stop_at_intersection_probability = None
        else:
            action_acceleration = action[0]
            action_indicator_probability = action[1]

        if self.use_das:
            speed_target = float(action_speed_target)
            A.driving_assistant_configure_speed_target(self.das, self.agent, self.sim, A.mps(speed_target))

            # Follow mode time-headway adjustment
            if self.follow_assist:
                new_thw_target = float(action_thw_target)
                A.driving_assistant_configure_thw(self.das, self.agent, self.sim, A.seconds(new_thw_target))

            # set stop at intersection
            if self.stop_assist:
                should_stop_at_intersection = abs(action_stop_at_intersection_probability) > 0.5 if self.deterministic_actions else self.np_random.random() < abs(action_stop_at_intersection_probability)
            else:
                should_stop_at_intersection = False

            cur_stop_intent = A.driving_assistant_get_should_stop_at_intersection(self.das)
            if (sit.is_an_intersection_upcoming or sit.is_approaching_dead_end) and sit.is_approaching_end_of_lane:
                # if already decided to stop previously, don't allow changing that decision until slow enough (stop or rolling stop)
                if bool(cur_stop_intent) and A.car_get_speed(self.agent) > A.from_mph(5):
                    should_stop_at_intersection = True
                    if self.verbose:
                        print("Locked stop at intersection intent to True until slow enough")

            A.driving_assistant_configure_should_stop_at_intersection(self.das, self.agent, self.sim, bool(should_stop_at_intersection))
        else:
            # decide acceleration. If travel direction and acceleration have the same sign, scale it so max_acc or max_acc_reverse_gear, depending on direction. If they have opposite signs, scale it to max_dec in the direction of the acceleration action, since we want to brake.
            max_acc = self.agent.capabilities.accel_profile.max_acceleration
            max_dec = self.agent.capabilities.accel_profile.max_deceleration
            max_acc_reverse_gear = self.agent.capabilities.accel_profile.max_acceleration_reverse_gear
            speed = A.car_get_speed(self.agent)

            acc_direction = 1 if action_acceleration >= 0 else -1
            acc_scale = 0
            if speed * action_acceleration >= 0:  # directions are the same. Continue to accelerate in the same direction
                acc_scale = max_acc if speed >= 0 else max_acc_reverse_gear
            else:   # braking
                acc_scale = max_dec

            apply_acceleration = acc_direction * acc_scale
            apply_acceleration = 0.95 * apply_acceleration + 0.05 * A.car_get_acceleration(self.agent)  # smooth it a bit
            A.car_set_acceleration(self.agent, apply_acceleration)

        # Merge/turn intent
        should_indicate = abs(action_indicator_probability) > 0.5 if self.deterministic_actions else self.np_random.random() < abs(action_indicator_probability)
        indicator_direction = (
            (A.INDICATOR_LEFT if action_indicator_probability < 0 else A.INDICATOR_RIGHT) if should_indicate else A.INDICATOR_NONE
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
            elif not (turn_makes_sense or merge_makes_sense):
                new_merge_intent = A.INDICATOR_NONE
                new_turn_intent = A.INDICATOR_NONE
                indicator_direction = A.INDICATOR_NONE
            else:   # both make sense
                new_merge_intent = indicator_direction

        if self.use_das:
            if sit.is_an_intersection_upcoming and sit.is_approaching_end_of_lane:
                # prevent last-moment lane changes or turn intent changes near intersections
                new_merge_intent = A.INDICATOR_NONE
                new_turn_intent = A.car_get_indicator_turn(self.agent)
                if self.verbose:
                    print(f"Locked turn intent = {'left' if new_turn_intent == A.INDICATOR_LEFT else ('right' if new_turn_intent == A.INDICATOR_RIGHT else 'none')} near intersection")
            A.driving_assistant_configure_merge_intent(self.das, self.agent, self.sim, new_merge_intent)
            A.driving_assistant_configure_turn_intent(self.das, self.agent, self.sim, new_turn_intent)
        else:
            A.car_set_indicator_turn_and_request(self.agent, new_turn_intent)
            A.car_set_indicator_lane_and_request(self.agent, new_merge_intent)

        if self.verbose:
            if self.use_das:
                print(
                    f"Action: Speed target: {A.to_mph(A.mps(speed_target)):.1f} mph, "
                    f"THW target: {action_thw_target:.2f} s, "
                    f"Indicator: {'off' if indicator_direction == A.INDICATOR_NONE else ('left' if indicator_direction == A.INDICATOR_LEFT else 'right')}, "
                    f"Stop at intersection: {should_stop_at_intersection}"
                )
            else:
                print(
                    f"Action: Acceleration: {action_acceleration:.2f}, "
                    f"Indicator: {'off' if indicator_direction == A.INDICATOR_NONE else ('left' if indicator_direction == A.INDICATOR_LEFT else 'right')}, "
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
        while not ((self.crash_ends_episode and crashed) or reached_goal or timeout or out_of_fuel) and elapsed_time < self.decision_interval - 1e-6:
            delta_time = min(self.TERMINATION_CHECK_INTERVAL, self.decision_interval - elapsed_time)    # make sure you don't end up simulating more than decision_interval if it is not a multiple of TERMINATION_CHECK_INTERVAL
            t0 = A.sim_get_time(self.sim)
            A.simulate(self.sim, delta_time)  # Advance simulation

            # Capture observation for framestacking
            current_frames = self._get_obs_as_list_of_images()
            processed_frames = [f.transpose(2, 0, 1).copy()
                                for f in current_frames]
            self.frame_buffer.append(processed_frames)

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
                # reward -= self.crash_penalty
                cost += self.cost_car
                crashed_forward = sit.is_vehicle_ahead and sit.distance_to_lead_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(sit.lead_vehicle) / 2) <= 0
                crashed_backward = sit.is_vehicle_behind and sit.distance_to_following_vehicle - (A.car_get_length(self.agent) / 2 + A.car_get_length(sit.following_vehicle) / 2) <= 0
                if sit.is_on_intersection:
                    crashed_on_intersection = True
            if reached_goal:
                reward += self.goal_reward
        terminated = (self.crash_ends_episode and crashed) or reached_goal or out_of_fuel or timeout

        if crashed:
            reward -= self.crash_penalty

        # shaped reward if goal lane is specified: reward for reducing manhat distance to goal
        if self.goal_lane is not None:
            prev_potential = -self.prev_opt_distance_to_goal    # type: ignore
            new_opt_distance_to_goal = self._get_opt_dist_to_goal()
            new_potential = -new_opt_distance_to_goal
            if reached_goal:
                new_potential = 0.0  # zero it out since we consider even entering the correct lane as reaching the goal (which is not always 0 manhat distance to the center of the lane)
            shaped_reward = self.reward_shaping_gamma * new_potential - prev_potential
            shaped_reward /= 10.0  # every 10 meters yeilds 1.0 reward
            if self.reward_shaping:
                reward += shaped_reward
            self.prev_opt_distance_to_goal = new_opt_distance_to_goal

        fuel_final = A.car_get_fuel_level(self.agent)
        fuel_consumed = fuel_initial - fuel_final
        fuel_consumed_gallons = A.to_gallons(fuel_consumed)
        if np.isfinite(fuel_consumed_gallons):
            cost += self.cost_gas_per_gallon * fuel_consumed_gallons

        is_late = A.sim_get_time(self.sim) > self.appointment_time

        # Start accumulating lost wage cost if appointment time has passed and goal lane is not reached yet
        if self.goal_lane is not None and not reached_goal and is_late:
            cost += self.cost_lost_wage_per_hour * self.decision_interval / 3600.0

        if self.may_lose_entire_day_wage and self.goal_lane is not None and not reached_goal and (timeout or out_of_fuel or (self.crash_ends_episode and crashed)):
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
            "aeb_in_progress_during_crash": float(self.das.aeb_in_progress) if crashed and self.use_das else np.nan,
            "reached_goal": reached_goal,
            "reached_in_time": reached_goal and not is_late,
            "reached_but_late": reached_goal and is_late,
            "out_of_fuel": out_of_fuel,
            "total_fuel_consumed": A.to_gallons(self.fuel_drive_beginning - fuel_final),
            "timeout": timeout,
            "cost": cost,
            "opt_dist_to_goal_m": self.prev_opt_distance_to_goal if self.goal_lane is not None else np.nan,
        }
        self.steps += 1
        self.synchronized_sim_speedup = self.sim.simulation_speedup  # record this so that we preserve it across resets if it was modified through GUI during the episode

        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Clean up simulation resources and disconnect from the rendering server.
        """
        A.sim_disconnect_from_render_server(self.sim)
        A.sim_free(self.sim)
        for cam in self.cams:
            A.rgbcam_free(cam)
        A.infos_display_free(self.infos_display)
        A.minimap_free(self.minimap)
        A.collisions_free(self.collision_checker)
        super().close()

    def render(self) -> np.ndarray:
        """
        Render the current simulation state to an RGB array.

        Returns:
            np.ndarray: Rendered RGB image of the current simulation state.
        """
        img = self.gridify(self._get_obs_as_list_of_images())

        return img

    def gridify(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Arrange a batch of images into a grid format for visualization.

        Args:
            images (list[np.ndarray]):List of images to be arranged in a grid.

        Returns:
            np.ndarray: Grid of images arranged in a square-like format.
        """
        N = len(images)
        H, W, C = images[0].shape
        grid_size = int(np.ceil(np.sqrt(N)))
        grid = np.zeros((grid_size * H, grid_size * W, C), dtype=images[0].dtype)

        for idx in range(N):
            row, col = divmod(idx, grid_size)
            grid[row * H:(row + 1) * H, col * W:(col + 1) * W, :] = images[idx]

        return grid
