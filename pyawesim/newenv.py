from typing import Any, List, Optional, Tuple, Dict, Union
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import bindings as A


# Constants
DEFAULT_RENDER_IP = "127.0.0.1"  # Default IP for rendering server
OBSERVATION_NORMALIZATION = {
    "speed": A.from_mph(100),  # Normalize speeds by 100 mph
    "acceleration": 12,  # Normalize accelerations by 12 m/s²
    "distance": 50,  # Normalize distances by 50 meters
    "buffer": 4.0,  # Normalize buffer distance by 4 meters
    "position": 500,  # Normalize coordinates by 500 meters (roughly half city width)
    "lanes": 10,  # Normalize lane counts by 10
    "countdown": 60,  # Normalize traffic light countdown by 60 seconds
    'time': 3600,  # Normalize time by 3600 seconds (1 hour)
    'fuel': 50,  # Normalize fuel level by 50 liters (~13.2 gallons)
    "thw": 5.0,  # Normalize time headway by 5 seconds
}


class AwesimCityEnv(gym.Env):

    def __init__(
        self,
        deprecated_map: bool = False,
        npc_rogue_factor: float = 0.05,
        perception_noise: bool = True,
        observation_radius: float = 200.0,
        cam_resolution: Tuple[int, int] = (128, 128),
        anti_aliasing: bool = True,
        reward_shaping: bool = False,
        reward_shaping_gamma: float = 1.0,
        reward_shaping_rew_per_meter: float = 0.01,
        city_width: int = 1500,
        dt: float = 0.02,
        delta_t: float = 0.1,
        decision_interval: float = 1.0,
        sim_duration: float = 3600.0,
        min_goal_manhat: Optional[Union[int, str]] = 500,  # travel at least this manhattan distance to goal lane from starting lane. The starting and goal lanes are always single lane roads.
        num_cars: int = 256,
        success_reward: float = 1.0,
        fail_reward: float = -1.0,
        crash_causes_fail: bool = True,
        init_fuel_gallons: float = 15 / 4.0,  # 15 gallons tank, start with 1/4 tank
        synchronized: bool = False,
        synchronized_sim_speedup: float = 1.0,
        should_render: bool = False,
        render_server_ip: str = DEFAULT_RENDER_IP,
        env_index: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")

        # Environment configuration
        self.deprecated_map = deprecated_map
        self.npc_rogue_factor = npc_rogue_factor
        self.cam_resolution = cam_resolution
        self.perception_noise = perception_noise
        self.observation_radius = observation_radius
        self.anti_aliasing = anti_aliasing
        self.reward_shaping = reward_shaping
        self.reward_shaping_gamma = reward_shaping_gamma
        self.city_width = city_width
        self.dt = dt
        self.delta_t = delta_t
        self.min_goal_manhat = min_goal_manhat
        self.num_cars = num_cars
        self.decision_interval = decision_interval
        self.sim_duration = sim_duration
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.crash_ends_episode = crash_causes_fail
        self.reward_shaping_rew_per_meter = reward_shaping_rew_per_meter
        self.init_fuel_gallons = init_fuel_gallons
        self.synchronized = synchronized
        self.synchronized_sim_speedup = synchronized_sim_speedup
        self.should_render = should_render
        self.render_server_ip = render_server_ip
        self.verbose = verbose
        self.render_mode = "rgb_array"
        print("seeding rng with ", env_index + 1)
        A.seed_rng(env_index + 1)  # Seed random number generator for reproducibility

        # Initialize simulation
        self.sim: A.Simulation = A.sim_malloc()  # Allocate simulation instance
        A.awesim_setup(self.sim, self.city_width, self.num_cars, self.dt, A.Monday_8_AM, A.WEATHER_SUNNY, self.deprecated_map)
        A.sim_set_agent_enabled(self.sim, True)  # Enable agent control
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)
        self.agent: A.Car = A.sim_get_agent_car(self.sim)  # Get agent vehicle
        self.situation: A.SituationalAwareness = A.sim_get_situational_awareness(self.sim, self.agent.id)
        self.das: A.DrivingAssistant = A.sim_get_driving_assistant(self.sim, self.agent.id)
        self.map: A.Map = A.sim_get_map(self.sim)
        self.cams: List[A.RGBCamera] = [A.rgbcam_malloc(A.coordinates_create(0, 0), 0, 0, cam_resolution[0], cam_resolution[1], A.from_degrees(0), A.meters(0)) for _ in range(7)]  # Initialize 7 cameras
        if self.anti_aliasing:
            for cam in self.cams:
                A.rgbcam_set_aa_level(cam, 2)  # 2x anti-aliasing
        self.infos_display: A.InfosDisplay = A.infos_display_malloc(cam_resolution[0], cam_resolution[1], len(self._get_info_for_info_display()))
        self.minimap: A.MiniMap = A.minimap_malloc(cam_resolution[0], cam_resolution[1], self.map)
        self.collision_checker: A.Collisions = A.collisions_malloc()
        self.path_planner: A.PathPlanner = A.path_planner_create(self.map, False)
        self.sample_random_goal()
        self.feature_dim = self._get_state().shape[1]

        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32 # shape: (num_entities, feature_dim)
        )

        # action space: discrete. Edit personality, indicate
        self.action_meanings = [
            # "set_style_no_rules",
            "set_style_reckless",
            "set_style_aggressive",
            "set_style_normal",
            "set_style_defensive",
            "indicate_left",
            "indicate_right",
            "indicate_nav",  # indicate based on navigation path
        ]
        self.action_meaning_to_driving_style = {
            "set_style_no_rules": A.SMART_DAS_DRIVING_STYLE_NO_RULES,
            "set_style_reckless": A.SMART_DAS_DRIVING_STYLE_RECKLESS,
            "set_style_aggressive": A.SMART_DAS_DRIVING_STYLE_AGGRESSIVE,
            "set_style_normal": A.SMART_DAS_DRIVING_STYLE_NORMAL,
            "set_style_defensive": A.SMART_DAS_DRIVING_STYLE_DEFENSIVE,
        }
        self.action_space = spaces.Discrete(len(self.action_meanings))
        self.action_space.names = self.action_meanings  # type: ignore

    def _is_start_road_valid(self, start_road: A.Road) -> bool:
        if A.road_get_num_lanes(start_road) != 1:       # must be a single lane road
            return False
        if A.road_get_type(start_road) != A.STRAIGHT:   # must not be a turn
            return False
        return True

    def _is_goal_valid(self, start_road: A.Road, goal_road: A.Road) -> bool:
        if A.road_get_num_lanes(goal_road) != 1:       # must be a single lane road
            return False
        if A.road_get_type(goal_road) != A.STRAIGHT:   # must not be a turn
            return False
        if A.lane_get_num_connections(A.road_get_rightmost_lane(goal_road, self.map)) == 0:  # dead-end road
            return False
        if A.lane_get_num_connections_incoming(A.road_get_rightmost_lane(goal_road, self.map)) == 0:  # unreachable road
            return False
        start_midpoint = A.road_get_mid_point(start_road)
        goal_midpoint = A.road_get_mid_point(goal_road)
        if A.vec_distance_manhat(start_midpoint, goal_midpoint) < self.min_goal_manhat:
            return False
        return True
        
    def _sample_goal_road(self, start_road: A.Road, max_tries: int) -> Optional[A.Road]:
        for _ in range(max_tries):
            goal_road_id = A.rand_int(A.map_get_num_roads(self.map))
            goal_road = A.map_get_road(self.map, goal_road_id)
            if self._is_goal_valid(start_road, goal_road):
                return goal_road
        return None

    def sample_random_goal(self):
        agent_lane = A.car_get_lane(self.agent, self.map)
        self.start_road: A.Road = A.map_get_road(self.map, agent_lane.road_id)
        if not self._is_start_road_valid(self.start_road):
            raise RuntimeError(f"Current start_road (id {self.start_road.id}) is not valid for sampling a goal.")
        goal_road = self._sample_goal_road(self.start_road, max_tries=100)
        if goal_road is None:
            raise RuntimeError("Failed to sample a valid goal road after maximum attempts.")
    
        self.goal_road: A.Road = goal_road
        self.start_lane: A.Lane = A.road_get_rightmost_lane(self.start_road, self.map)
        self.goal_lane: A.Lane = A.road_get_rightmost_lane(self.goal_road, self.map)
        self.goal_point: A.Vec2D = A.lane_get_mid_point(self.goal_lane)

    def _get_state(self) -> np.ndarray:
        A.situational_awareness_build(self.sim, self.agent.id)
        cars_matrix = self._get_cars_matrix()  # (max_cars, car_feature_dim)
        roads_matrix = self._get_roads_matrix()  # (max_roads, road_feature_dim)
        intersections_matrix = self._get_intersections_matrix()  # (max_intersections, intersection_feature_dim)
        misc_state_matrix = self._get_misc_state_matrix()  # (1, misc_feature_dim)
        feature_dim = max(cars_matrix.shape[1], roads_matrix.shape[1], intersections_matrix.shape[1], misc_state_matrix.shape[1])
        # pad others:
        def pad_matrix(mat: np.ndarray, target_dim: int) -> np.ndarray:
            if mat.shape[1] < target_dim:
                padding = np.zeros((mat.shape[0], target_dim - mat.shape[1]), dtype=np.float32)
                mat = np.concatenate([mat, padding], axis=1)
            return mat
        # pad all to feature_dim
        cars_matrix = pad_matrix(cars_matrix, feature_dim)
        roads_matrix = pad_matrix(roads_matrix, feature_dim)
        intersections_matrix = pad_matrix(intersections_matrix, feature_dim)
        misc_state_matrix = pad_matrix(misc_state_matrix, feature_dim)
        state = np.concatenate([cars_matrix, roads_matrix, intersections_matrix, misc_state_matrix], axis=0)  # (num_entities, feature_dim)

        self.car_feature_dim = cars_matrix.shape[1]
        self.car_count = cars_matrix.shape[0]
        self.road_feature_dim = roads_matrix.shape[1]
        self.road_count = roads_matrix.shape[0]
        self.intersection_feature_dim = intersections_matrix.shape[1]
        self.intersection_count = intersections_matrix.shape[0]
        self.misc_state_feature_dim = misc_state_matrix.shape[1]
        self.misc_state_entity_count = misc_state_matrix.shape[0]
        self.entity_count = state.shape[0]
        return state
    
    def _get_cars_matrix_from_state(self, state):
        # extract cars from last two dims of state (which may have a batch dim)
        return state[..., :self.car_count, :self.car_feature_dim]
    
    def _get_roads_matrix_from_state(self, state):
        return state[..., self.car_count:self.car_count + self.road_count, :self.road_feature_dim]
    
    def _get_intersections_matrix_from_state(self, state):
        return state[..., self.car_count + self.road_count:self.car_count + self.road_count + self.intersection_count, :self.intersection_feature_dim]
    
    def _get_misc_state_matrix_from_state(self, state):
        return state[..., -self.misc_state_entity_count:, :self.misc_state_feature_dim]

    
    def _get_misc_state_matrix(self) -> np.ndarray:
        # goal, fuel, adas state
        goal = A.vec_scale(self.goal_point, 1 / OBSERVATION_NORMALIZATION["position"])
        fuel = A.car_get_fuel_level(self.agent) / OBSERVATION_NORMALIZATION["fuel"]
        speed_target = self.das.speed_target / OBSERVATION_NORMALIZATION["speed"]
        speed_limit = self.situation.lane.speed_limit / OBSERVATION_NORMALIZATION["speed"]
        thw = self.das.thw / OBSERVATION_NORMALIZATION["thw"]
        buffer = self.das.buffer / OBSERVATION_NORMALIZATION["buffer"]
        aeb_in_progress = float(self.das.aeb_in_progress)
        intersection_approach_engaged = float(self.das.smart_das_intersection_approach_engaged)
        use_preferred_accel_profile = float(self.das.use_preferred_accel_profile)
        use_linear_speed_control = float(self.das.use_linear_speed_control)
        is_style_no_rules = float(self.das.smart_das_driving_style == A.SMART_DAS_DRIVING_STYLE_NO_RULES)
        is_style_reckless = float(self.das.smart_das_driving_style == A.SMART_DAS_DRIVING_STYLE_RECKLESS)
        is_style_aggressive = float(self.das.smart_das_driving_style == A.SMART_DAS_DRIVING_STYLE_AGGRESSIVE)
        is_style_normal = float(self.das.smart_das_driving_style == A.SMART_DAS_DRIVING_STYLE_NORMAL)
        is_style_defensive = float(self.das.smart_das_driving_style == A.SMART_DAS_DRIVING_STYLE_DEFENSIVE)
        nav_ac_0 = self._get_path_planner_indicating_recommendation(skip=0) - 1
        nav_ac_1 = self._get_path_planner_indicating_recommendation(skip=1) - 1
        nav_ac_2 = self._get_path_planner_indicating_recommendation(skip=2) - 1
        nav_ac_3 = self._get_path_planner_indicating_recommendation(skip=3) - 1

        state_vector = np.array([
            goal.x, goal.y,
            fuel,
            speed_target,
            speed_limit,
            thw,
            buffer,
            aeb_in_progress,
            intersection_approach_engaged,
            use_preferred_accel_profile,
            use_linear_speed_control,
            is_style_no_rules,
            is_style_reckless,
            is_style_aggressive,
            is_style_normal,
            is_style_defensive,
            nav_ac_0, nav_ac_1, nav_ac_2, nav_ac_3,
        ], dtype=np.float32)
        state_vector = state_vector.reshape(1, -1)  # (1, misc_feature_dim)

        return state_vector

    def _get_car_state(self, car: A.Car, relative_to_agent: bool = False) -> np.ndarray:
        if car is None:
            return np.zeros((self.car_feature_dim,), dtype=np.float32)
    
        A.situational_awareness_build(self.sim, car.id)
        A.situational_awareness_build(self.sim, self.agent.id)

        sa = A.sim_get_situational_awareness(self.sim, car.id)
    
        exists = True
        if relative_to_agent:
            position = A.coords_compute_relative_to_car(self.agent, car.center)
            if A.vec_magnitude(position) > self.observation_radius:
                exists = False
                return np.zeros((23,), dtype=np.float32)
            speed = A.vec2d_rotate_to_car_frame(self.agent, A.vec_sub(car.true_speed_vector, self.agent.true_speed_vector))
            acc = A.vec2d_rotate_to_car_frame(self.agent, A.vec_sub(car.true_acceleration_vector, self.agent.true_acceleration_vector))
            orien = car.orientation - self.agent.orientation
            orien_cos, orien_sin = np.cos(orien), np.sin(orien)
        else:
            position = car.center
            speed = car.true_speed_vector
            acc = car.true_acceleration_vector
            orien_cos, orien_sin = np.cos(car.orientation), np.sin(car.orientation)

        is_approaching_intersection = sa.is_an_intersection_upcoming
        is_on_intersection = sa.is_on_intersection
        lane_no = -1
        num_lanes = 0
        road_center = A.coordinates_create(0, 0)
        intersection_center = A.coordinates_create(0, 0)
        
        if sa.road is not None:
            lane_no = A.road_find_index_of_lane(sa.road, sa.lane.id)
            num_lanes = A.road_get_num_lanes(sa.road)
            road_center = A.road_get_mid_point(sa.road)
            intersection_center = A.coordinates_create(0, 0)
        if sa.intersection is not None:
            lane_no = -1  # not on a road lane
            num_lanes = A.intersection_get_num_lanes(sa.intersection)
            road_center = A.coordinates_create(0, 0)
            intersection_center = A.intersection_get_center(sa.intersection)

        width = car.dimensions.x
        length = car.dimensions.y
        ind_left = car.indicator_lane == A.INDICATOR_LEFT or car.indicator_turn == A.INDICATOR_LEFT
        ind_none = car.indicator_lane == A.INDICATOR_NONE and car.indicator_turn == A.INDICATOR_NONE
        ind_right = car.indicator_lane == A.INDICATOR_RIGHT or car.indicator_turn == A.INDICATOR_RIGHT
        my_turn_stop_intersection = sa.is_my_turn_at_stop_sign_fcfs

        # normalize
        scale_pos = 1 / OBSERVATION_NORMALIZATION["distance"] if relative_to_agent else 1 / OBSERVATION_NORMALIZATION["position"]
        position = A.vec_scale(position, scale_pos)
        scale_speed = 0.1 / OBSERVATION_NORMALIZATION["speed"] if relative_to_agent else 1 / OBSERVATION_NORMALIZATION["speed"]
        speed = A.vec_scale(speed, scale_speed)
        acc = A.vec_scale(acc, 1 / OBSERVATION_NORMALIZATION["acceleration"])
        lane_no /= OBSERVATION_NORMALIZATION["lanes"]
        num_lanes /= OBSERVATION_NORMALIZATION["lanes"]
        road_center = A.vec_scale(road_center, 1 / OBSERVATION_NORMALIZATION["position"])
        intersection_center = A.vec_scale(intersection_center, 1 / OBSERVATION_NORMALIZATION["position"])
        width /= 2.43       # (8 ft in meters)
        length /= 6.096     # (20 ft in meters)

        state_vector = np.array([
            float(exists),
            position.x, position.y,
            speed.x, speed.y,
            acc.x, acc.y,
            orien_sin, orien_cos,
            lane_no,
            num_lanes,
            float(is_approaching_intersection), float(is_on_intersection),
            road_center.x, road_center.y,
            intersection_center.x, intersection_center.y,
            width, length,
            float(ind_left), float(ind_none), float(ind_right),
            float(my_turn_stop_intersection),
        ], dtype=np.float32)

        return state_vector

    def _get_cars_matrix(self) -> np.ndarray:
        cars_states = [self._get_car_state(A.sim_get_car(self.sim, car_id), relative_to_agent=(car_id != self.agent.id)) for car_id in range(self.num_cars)]
        cars_matrix = np.stack(cars_states, axis=0)  # (max_cars, feature_dim)
        return cars_matrix

    def _get_road_state(self, road_id) -> np.ndarray:
        road = A.map_get_road(self.map, road_id)
        num_lanes = A.road_get_num_lanes(road)
        speed_limit = A.road_get_speed_limit(road)
        is_E = road.direction == A.DIRECTION_EAST
        is_W = road.direction == A.DIRECTION_WEST
        is_N = road.direction == A.DIRECTION_NORTH
        is_S = road.direction == A.DIRECTION_SOUTH
        is_CCW = road.direction == A.DIRECTION_CCW
        is_CW = road.direction == A.DIRECTION_CW
        start_x = road.start_point.x
        start_y = road.start_point.y
        midpoint_x = road.mid_point.x
        midpoint_y = road.mid_point.y
        end_x = road.end_point.x
        end_y = road.end_point.y

        # normalize
        num_lanes /= OBSERVATION_NORMALIZATION["lanes"]
        speed_limit /= OBSERVATION_NORMALIZATION["speed"]
        start_x /= OBSERVATION_NORMALIZATION["position"]
        start_y /= OBSERVATION_NORMALIZATION["position"]
        midpoint_x /= OBSERVATION_NORMALIZATION["position"]
        midpoint_y /= OBSERVATION_NORMALIZATION["position"]
        end_x /= OBSERVATION_NORMALIZATION["position"]
        end_y /= OBSERVATION_NORMALIZATION["position"]

        state_vector = np.array([
            num_lanes, speed_limit,
            float(is_E), float(is_W), float(is_N), float(is_S), float(is_CCW), float(is_CW),
            start_x, start_y,
            midpoint_x, midpoint_y,
            end_x, end_y,
        ], dtype=np.float32)

        return state_vector

    def _get_roads_matrix(self) -> np.ndarray:
        road_states = [self._get_road_state(road_id) for road_id in range(A.map_get_num_roads(self.map))]
        roads_matrix = np.stack(road_states, axis=0)  # (max_roads, feature_dim)
        return roads_matrix
    
    def _get_intersection_state(self, intersection_id) -> np.ndarray:
        intersection = A.map_get_intersection(self.map, intersection_id)
        intersection_center = intersection.center
        intersection_state_is_NS_green_EW_red = intersection.state == A.NS_GREEN_EW_RED
        intersection_state_is_NS_YELLOW_EW_RED = intersection.state == A.NS_YELLOW_EW_RED
        intersection_state_is_ALL_RED_BEFORE_EW_GREEN = intersection.state == A.ALL_RED_BEFORE_EW_GREEN
        intersection_state_is_NS_RED_EW_GREEN = intersection.state == A.NS_RED_EW_GREEN
        intersection_state_is_EW_YELLOW_NS_RED = intersection.state == A.EW_YELLOW_NS_RED
        intersection_state_is_ALL_RED_BEFORE_NS_GREEN = intersection.state == A.ALL_RED_BEFORE_NS_GREEN
        intersection_state_is_FOUR_WAY_STOP = intersection.state == A.FOUR_WAY_STOP
        intersection_state_is_T_JUNC_NS = intersection.state == A.T_JUNC_NS
        intersection_state_is_T_JUNC_EW = intersection.state == A.T_JUNC_EW

        # normalize
        intersection_center = A.vec_scale(intersection_center, 1 / OBSERVATION_NORMALIZATION["position"])

        return np.array([
            intersection_center.x, intersection_center.y,
            float(intersection_state_is_NS_green_EW_red),
            float(intersection_state_is_NS_YELLOW_EW_RED),
            float(intersection_state_is_ALL_RED_BEFORE_EW_GREEN),
            float(intersection_state_is_NS_RED_EW_GREEN),
            float(intersection_state_is_EW_YELLOW_NS_RED),
            float(intersection_state_is_ALL_RED_BEFORE_NS_GREEN),
            float(intersection_state_is_FOUR_WAY_STOP),
            float(intersection_state_is_T_JUNC_NS),
            float(intersection_state_is_T_JUNC_EW),
        ], dtype=np.float32)
    
    def _get_intersections_matrix(self) -> np.ndarray:
        intersection_states = [self._get_intersection_state(intersection_id) for intersection_id in range(A.map_get_num_intersections(self.map))]
        intersections_matrix = np.stack(intersection_states, axis=0)  # (max_intersections, feature_dim)
        return intersections_matrix
    
    def _get_path_planner_opt_dist_to_goal(self) -> float:
        dist = float(self.path_planner.optimal_cost)
        dist = min(dist, self.city_width * 10)  # cap to avoid numerical issues
        return dist


    def _nav_action_to_indicator(self, nav_action: int) -> int:
        if nav_action in (A.NAV_TURN_LEFT, A.NAV_LANE_CHANGE_LEFT, A.NAV_MERGE_LEFT):
            return A.INDICATOR_LEFT
        elif nav_action in (A.NAV_TURN_RIGHT, A.NAV_LANE_CHANGE_RIGHT, A.NAV_MERGE_RIGHT):
            return A.INDICATOR_RIGHT
        else:
            return A.INDICATOR_NONE
        
    def _get_path_planner_indicating_recommendation(self, skip: int = 0) -> int:
        nav_action = A.path_planner_get_solution_action(self.path_planner, True, skip)
        return self._nav_action_to_indicator(nav_action)
        
    def _get_current_indicator(self) -> int:
        cur_ind_left = self.agent.indicator_lane == A.INDICATOR_LEFT or self.agent.indicator_turn == A.INDICATOR_LEFT
        cur_ind_none = self.agent.indicator_lane == A.INDICATOR_NONE and self.agent.indicator_turn == A.INDICATOR_NONE
        cur_ind_right = self.agent.indicator_lane == A.INDICATOR_RIGHT or self.agent.indicator_turn == A.INDICATOR_RIGHT
        if cur_ind_left:
            return A.INDICATOR_LEFT
        elif cur_ind_right:
            return A.INDICATOR_RIGHT
        else:
            return A.INDICATOR_NONE


    def _get_info_for_info_display(self) -> list[Tuple[float, A.RGB]]:
        A.situational_awareness_build(self.sim, self.agent.id)
        das = self.das
        sit = self.situation

        # car state
        fuel = A.car_get_fuel_level(self.agent) / OBSERVATION_NORMALIZATION["fuel"]
        speed = A.car_get_speed(self.agent) / OBSERVATION_NORMALIZATION["speed"]
        accel = A.car_get_acceleration(self.agent) / OBSERVATION_NORMALIZATION["acceleration"]
        ind_turn_left = A.car_get_indicator_turn(self.agent) == A.INDICATOR_LEFT
        ind_merge_left = A.car_get_indicator_lane(self.agent) == A.INDICATOR_LEFT
        ind_left = ind_turn_left or ind_merge_left
        ind_turn_right = A.car_get_indicator_turn(self.agent) == A.INDICATOR_RIGHT
        ind_merge_right = A.car_get_indicator_lane(self.agent) == A.INDICATOR_RIGHT
        ind_right = ind_turn_right or ind_merge_right

        # ADAS state
        speed_target = das.speed_target / OBSERVATION_NORMALIZATION["speed"]
        thw_target = das.thw / OBSERVATION_NORMALIZATION["thw"]
        intersection_approach_engaged = das.smart_das_intersection_approach_engaged
        aeb_in_progress = das.aeb_in_progress
        defensiveness = das.smart_das_driving_style / (A.SMART_DAS_DRIVING_STYLES_COUNT - 1)

        # environment state
        # t = A.sim_get_time(self.sim) / OBSERVATION_NORMALIZATION["time"]
        my_turn_fcfs = float(sit.is_my_turn_at_stop_sign_fcfs)
        speed_limit = sit.lane.speed_limit / OBSERVATION_NORMALIZATION["speed"]
        

        infos_and_colors = [
            (fuel, A.RGB_WHITE),
            (abs(speed), A.RGB_BLUE if speed >= 0 else A.RGB_YELLOW),
            (abs(accel), A.RGB_GREEN if accel >= 0 else A.RGB_RED),

            (float(ind_left), A.RGB_ORANGE),
            (abs(speed_target), A.RGB_MAGENTA if speed_target >= 0 else A.RGB_YELLOW),
            (float(ind_right), A.RGB_ORANGE),

            (float(intersection_approach_engaged), A.RGB_RED),
            (thw_target, A.RGB_CYAN),
            (float(aeb_in_progress), A.RGB_RED),

            (my_turn_fcfs, A.RGB_GREEN),
            (defensiveness, A.RGB_BLUE),
            (speed_limit, A.RGB_WHITE),
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
        A.path_planner_compute_shortest_path(self.path_planner, self.situation.lane, self.situation.lane_progress_m, self.goal_lane, 0.5 * self.goal_lane.length)
        self.minimap.marked_path = self.path_planner
        A.minimap_render(self.minimap, self.sim)

        # info display render
        infos_and_colors = self._get_info_for_info_display()
        for i, (info, color) in enumerate(infos_and_colors):
            A.infos_display_set_info(self.infos_display, i, info)
            A.infos_display_set_info_color(self.infos_display, i, color)
        A.infos_display_render(self.infos_display)

        return [self.cams[0].to_numpy(), self.cams[1].to_numpy(), self.cams[2].to_numpy(), self.cams[3].to_numpy(), self.cams[4].to_numpy(), self.cams[5].to_numpy(), self.cams[6].to_numpy(), self.minimap.to_numpy(), self.infos_display.to_numpy()]   # type: ignore

    def _get_observation(self) -> np.ndarray:
        return self._get_state()

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
        return self.situation.lane.id == self.goal_lane.id and self.situation.lane_progress > 0.1

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

        num_cars = max(np.random.randint(self.num_cars // 4, self.num_cars + 1), 16)    # randomize number of cars a bit. At least 16 cars.
        if self.num_cars == 1:
            num_cars = 1    # for debugging
        A.awesim_setup(self.sim, self.city_width, num_cars, 0.02, A.Monday_8_AM, A.WEATHER_SUNNY, self.deprecated_map)
        # Configure simulation settings
        if self.synchronized:
            A.sim_set_synchronized(self.sim, True, self.synchronized_sim_speedup)
        if self.should_render:
            A.sim_connect_to_render_server(self.sim, self.render_server_ip, 4242)

        A.sim_set_npc_rogue_factor(self.sim, self.npc_rogue_factor)
        if self.perception_noise:
            self.sim.perception_noise_distance_std_dev_percent = 0.05; # 5% distance error
            self.sim.perception_noise_speed_std_dev = A.from_mph(0.5); # 0.5 mph speed error
            self.sim.perception_noise_dropout_probability = 0.01; # 1% dropout probability
            self.sim.perception_noise_blind_spot_dropout_base_probability = 0.1; # 5% blind spot dropout probabilityelative positions. The base is the multiplier)
        else:
            self.sim.perception_noise_distance_std_dev_percent = 0.0
            self.sim.perception_noise_speed_std_dev = 0.0
            self.sim.perception_noise_dropout_probability = 0.0
            self.sim.perception_noise_blind_spot_dropout_base_probability = 0.0

        A.sim_set_agent_enabled(self.sim, True)
        A.sim_set_agent_driving_assistant_enabled(self.sim, True)
        A.situational_awareness_build(self.sim, self.agent.id)
        A.driving_assistant_reset_settings(self.das, self.agent)
        A.driving_assistant_configure_smart_das(self.das, self.agent, self.sim, True)
        A.driving_assistant_configure_smart_das_driving_style(self.das, self.agent, self.sim, A.SMART_DAS_DRIVING_STYLE_NORMAL)
        A.driving_assistant_configure_aeb_assistance(self.das, self.agent, self.sim, True)
        A.driving_assistant_smart_update_das_variables(self.das, self.agent, self.sim, self.situation)

        self.sample_random_goal()
        A.path_planner_compute_shortest_path(self.path_planner, self.situation.lane, self.situation.lane_progress_m, self.goal_lane, 0.5 * self.goal_lane.length)
        self.prev_opt_distance_to_goal = self._get_path_planner_opt_dist_to_goal()
        self.sim.agent_goal_lane_id = self.goal_lane.id

        # Configure collision checker
        A.collisions_reset(self.collision_checker)

        # Configure fuel
        A.car_set_fuel_level(self.agent, A.from_gallons(self.init_fuel_gallons))
        A.car_set_fuel_tank_capacity(self.agent, A.from_gallons(self.init_fuel_gallons))    # unnecessary but just in case as default tank capacity is Inf for all cars in awesim

        self.steps = 0
        self.fuel_drive_beginning = A.car_get_fuel_level(self.agent)

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        A.situational_awareness_build(self.sim, self.agent.id)
        A.path_planner_compute_shortest_path(self.path_planner, self.situation.lane, self.situation.lane_progress_m, self.goal_lane, 0.5 * self.goal_lane.length)
        sit = self.situation

        # action = self.action_space.n - 1    # for debugging: always indicate nav

        if action < 0 or action >= len(self.action_meanings):
            raise ValueError(f"Invalid action {action}. Must be in [0, {len(self.action_meanings) - 1}].")
        
        if self.action_meanings[action] in self.action_meaning_to_driving_style.keys():
            new_driving_style = self.action_meaning_to_driving_style[self.action_meanings[action]]
            A.driving_assistant_configure_smart_das_driving_style(self.das, self.agent, self.sim, new_driving_style)
            # indicator is preserved
        else:
            cur_ind = self._get_current_indicator()
            cancel_indicate = self.action_meanings[action] == "indicate_left" and cur_ind == A.INDICATOR_LEFT or self.action_meanings[action] == "indicate_right" and cur_ind == A.INDICATOR_RIGHT

            indicator_directions = {
                "indicate_left": A.INDICATOR_LEFT,
                "indicate_right": A.INDICATOR_RIGHT,
                "indicate_nav": self._get_path_planner_indicating_recommendation(),
            }

            indicator_direction = indicator_directions[self.action_meanings[action]] if not cancel_indicate else A.INDICATOR_NONE
            # print(f"Requested indicator direction: {indicator_direction}")
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
                # print(f"turn_makes_sense: {turn_makes_sense}, merge_makes_sense: {merge_makes_sense}")
                if turn_makes_sense and not merge_makes_sense:
                    new_turn_intent = indicator_direction
                    # print(f"Setting turn intent to {indicator_direction}")
                elif merge_makes_sense and not turn_makes_sense:
                    new_merge_intent = indicator_direction
                    # print(f"Setting merge intent to {indicator_direction}")
                elif not (turn_makes_sense or merge_makes_sense):
                    new_merge_intent = A.INDICATOR_NONE
                    new_turn_intent = A.INDICATOR_NONE
                    indicator_direction = A.INDICATOR_NONE
                    # print("Neither turn nor merge make sense. Setting both intents to NONE.")
                else:   # both make sense, prefer merge
                    new_merge_intent = indicator_direction
                    # print(f"Both turn and merge make sense. Setting merge intent to {indicator_direction}.")

            # print(f"Action: {self.action_meanings[action]}, Setting indicator to {indicator_direction}, turn_intent: {new_turn_intent}, merge_intent: {new_merge_intent}")
            A.driving_assistant_configure_merge_intent(self.das, self.agent, self.sim, new_merge_intent)
            A.driving_assistant_configure_turn_intent(self.das, self.agent, self.sim, new_turn_intent)

        # Simulate and compute reward
        A.collisions_reset(self.collision_checker)
        reward = 0.0
        cost = 0.0
        elapsed_time = 0.0
        sim_time_initial = A.sim_get_time(self.sim)
        crashed = reached_goal = timeout = out_of_fuel = False
        crashed_forward = crashed_backward = False
        crashed_on_intersection = False
        fuel_initial = A.car_get_fuel_level(self.agent)
        # print(f"Beginning step simulation loop. decision_interval: {self.decision_interval}s")
        while not ((self.crash_ends_episode and crashed) or reached_goal or timeout or out_of_fuel) and elapsed_time < self.decision_interval - 1e-6:
            delta_time = min(self.delta_t, self.decision_interval - elapsed_time)    # make sure you don't end up simulating more than decision_interval if it is not a multiple of TERMINATION_CHECK_INTERVAL
            t0 = A.sim_get_time(self.sim)
            A.simulate(self.sim, delta_time)  # Advance simulation

            delta_time = A.sim_get_time(self.sim) - t0  # true delta time (since it is not guaranteed that sim would have simulated exactly for delta_time if it is not a multiple of dt.)
            elapsed_time = A.sim_get_time(self.sim) - sim_time_initial
            A.situational_awareness_build(self.sim, self.agent.id)
            crashed = self._crashed()
            reached_goal = self._reached_destination()
            timeout = A.sim_get_time(self.sim) >= self.sim_duration
            out_of_fuel = A.car_get_fuel_level(self.agent) < 1e-9
            if crashed:
                crashed_forward = False  # todo: implement
                crashed_backward = False  # todo: implement
                if sit.is_on_intersection:
                    crashed_on_intersection = True
            if reached_goal:
                reward += self.success_reward
        # print(f"Finished step simulation loop. Elapsed time: {elapsed_time}s")
        terminated = (self.crash_ends_episode and crashed) or reached_goal or out_of_fuel or timeout

        if terminated and not reached_goal:
            reward += self.fail_reward

        # shaped reward if goal lane is specified: reward for reducing manhat distance to goal
        if self.goal_lane is not None:
            prev_potential = -self.prev_opt_distance_to_goal    # type: ignore
            new_opt_distance_to_goal = self._get_path_planner_opt_dist_to_goal()
            new_potential = -new_opt_distance_to_goal
            if reached_goal:
                new_potential = 0.0  # zero it out since we consider even entering the correct lane as reaching the goal (which is not always 0 manhat distance to the center of the lane)
            shaped_reward = self.reward_shaping_gamma * new_potential - prev_potential
            shaped_reward *= self.reward_shaping_rew_per_meter
            if self.reward_shaping:
                reward += shaped_reward
            self.prev_opt_distance_to_goal = new_opt_distance_to_goal

        fuel_final = A.car_get_fuel_level(self.agent)
        # fuel_consumed = fuel_initial - fuel_final


        if self.verbose:
            if crashed:
                print("Crashed!")
            if reached_goal:
                print("Reached goal!")
            if out_of_fuel:
                print("Out of fuel!")
            if timeout:
                print("Timed out!")

        A.situational_awareness_build(self.sim, self.agent.id)
        A.path_planner_compute_shortest_path(self.path_planner, self.situation.lane, self.situation.lane_progress_m, self.goal_lane, 0.5 * self.goal_lane.length)
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
            "out_of_fuel": out_of_fuel,
            "total_fuel_consumed": A.to_gallons(self.fuel_drive_beginning - fuel_final),
            "timeout": timeout,
            "cost": cost,
            "opt_dist_to_goal_m": self.prev_opt_distance_to_goal if self.goal_lane is not None else np.nan,
            "action_no_rules": self.action_meanings[action] == "set_style_no_rules",
            "action_reckless": self.action_meanings[action] == "set_style_reckless",
            "action_aggressive": self.action_meanings[action] == "set_style_aggressive",
            "action_normal": self.action_meanings[action] == "set_style_normal",
            "action_defensive": self.action_meanings[action] == "set_style_defensive",
            "action_indicate_left": self.action_meanings[action] == "indicate_left",
            "action_indicate_right": self.action_meanings[action] == "indicate_right",
            "action_indicate_nav": self.action_meanings[action] == "indicate_nav",
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
