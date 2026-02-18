#include "ai.h"
#include "map.h"
#include "sim.h"
#include "car.h"
#include "utils.h"
#include "logging.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

// Constants for graph generation
#define LANE_CHANGE_INTERVAL 40.0 // Meters between lane change nodes

// Traffic flow smoothing: exponential moving average
// alpha = dt / tau, where tau is the effective time constant.
// For a ~5 minute window with updates every 1s: tau = 300s, alpha ~ 0.0033
#define TRAFFIC_FLOW_TAU 60.0                  // EMA time constant in seconds (effective ~1 min window)
#define TRAFFIC_FLOW_MIN_SPEED 0.044704              // Minimum traffic flow speed (m/s) to avoid division by zero in cost
#define HIGHWAY_AVOIDANCE_PENALTY 10.0          // Cost multiplier for highway lanes (roads with 3+ lanes) when avoid_highways is enabled


static Vec2D get_node_pos(const PathPlanner* planner, int node_idx) {
    if (node_idx < 0 || node_idx >= MAX_GRAPH_NODES) return vec_create(0, 0);
    const MapNode* node = &planner->map_nodes[node_idx];
    Lane* lane = map_get_lane(planner->map, node->lane_id);
    if (!lane) return vec_create(0, 0);

    // Calculate position based on progress
    if (lane->type == LINEAR_LANE) {
        Vec2D diff = vec_sub(lane->end_point, lane->start_point);
        double len = lane->length;
        if (len < 1e-3) return lane->start_point;
        double t = node->progress / len;
        // Clamp t [0, 1]
        if (t < 0) t = 0; if (t > 1) t = 1;
        return vec_add(lane->start_point, vec_scale(diff, t));
    } else {
        // Quarter Arc
        double angle_diff = lane->end_angle - lane->start_angle;
        double len = lane->length;
         if (len < 1e-3) return lane->start_point;
        double t = node->progress / len; 
         // Clamp t [0, 1]
        if (t < 0) t = 0; if (t > 1) t = 1;
        
        double angle = lane->start_angle + angle_diff * t;
        return vec_create(
            lane->center.x + lane->radius * cos(angle),
            lane->center.y + lane->radius * sin(angle)
        );
    }
}

static double astar_heuristic(int node_idx, int end_idx, void* user_data) {
    PathPlanner* planner = (PathPlanner*)user_data;
    Vec2D p1 = get_node_pos(planner, node_idx);
    Vec2D p2 = get_node_pos(planner, end_idx);
    
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx*dx + dy*dy);
}


static double get_cost_with_traffic(double distance, double speed_limit, bool consider_speed_limits, bool use_live_traffic, double traffic_flow);

void path_planner_build_graph(PathPlanner* planner) {
    if (!planner->map || !planner->decision_graph) return;

    // Pre-pass: Identify extra points needed on lanes due to merges from other lanes
    // We use a simple array of arrays. Since we don't know how many points, we'll use a fixed max.
    // Assuming max 8 extra points per lane from merges is sufficient.
    #define MAX_EXTRA_POINTS 8
    double extra_points[MAX_NUM_LANES][MAX_EXTRA_POINTS];
    int extra_counts[MAX_NUM_LANES];
    memset(extra_counts, 0, sizeof(extra_counts));

    for (int i = 0; i < planner->map->num_lanes; ++i) {
        Lane* lane = map_get_lane(planner->map, i);
        if (!lane) continue;

        if (lane->merges_into_id != ID_NULL) {
            LaneId target_id = lane->merges_into_id;
            Lane* target = map_get_lane(planner->map, target_id);
            if (target) {
                // The end of the entry lane corresponds to this position on the target lane
                double end_proj = lane->merges_into_start * target->length + lane->length;
                
                if (extra_counts[target_id] < MAX_EXTRA_POINTS) {
                    extra_points[target_id][extra_counts[target_id]++] = end_proj;
                }
            }
        }
    }
    
    // 1. Generate Nodes
    for (int i = 0; i < planner->map->num_lanes; ++i) {
        Lane* lane = map_get_lane(planner->map, i);
        if (!lane) continue;

        // Use a temporary array to collect progress points to sort/dedup
        double points[MAX_NODES_PER_LANE];
        int count = 0;

        // Helper to add point
        #define ADD_POINT(p) { \
            if (count < MAX_NODES_PER_LANE) { \
                points[count++] = (p); \
            } \
        }

        ADD_POINT(0.0);
        ADD_POINT(lane->length);

        // Add extra points from pre-pass
        for (int k = 0; k < extra_counts[i]; ++k) {
            ADD_POINT(extra_points[i][k]);
        }

        for (double p = LANE_CHANGE_INTERVAL; p < lane->length; p += LANE_CHANGE_INTERVAL) {
            ADD_POINT(p);
        }

        // Merge/Exit points alignment
        if (lane->merges_into_id != ID_NULL) {
            LaneId target_id = lane->merges_into_id;
            Lane* target = map_get_lane(planner->map, target_id);
            
            double offset = lane->merges_into_start * target->length;
            double target_len = target->length;
            
            for (double t = 0; t <= target_len; t += LANE_CHANGE_INTERVAL) {
                double p = t - offset;
                if (p >= 0 && p <= lane->length) {
                    ADD_POINT(p);
                }
            }
            double p_end = target_len - offset;
            if (p_end >= 0 && p_end <= lane->length) ADD_POINT(p_end);
        }
        
        if (lane->exit_lane_id != ID_NULL) {
            LaneId target_id = lane->exit_lane_id;
            Lane* target = map_get_lane(planner->map, target_id);
            
            double exit_start = lane->exit_lane_start * lane->length;
            double exit_end = lane->exit_lane_end * lane->length;
            
            for (double t = 0; t <= target->length; t += LANE_CHANGE_INTERVAL) {
                double p = exit_start + t;
                if (p >= exit_start && p <= exit_end) {
                    ADD_POINT(p);
                }
            }
            ADD_POINT(exit_start);
            ADD_POINT(exit_end);
        }

        // Sort points
        for(int a=0; a<count-1; ++a) {
            for(int b=0; b<count-a-1; ++b) {
                if(points[b] > points[b+1]) {
                    double temp = points[b];
                    points[b] = points[b+1];
                    points[b+1] = temp;
                }
            }
        }

        // Add to graph and lookup tables
        double last_p = -1.0;
        for(int j=0; j<count; ++j) {
            double p = points[j];
            if (p < 0) p = 0;
            if (p > lane->length) p = lane->length;
            
            if (fabs(p - last_p) < 0.1) continue; // Dedup
            last_p = p;

            int node_id = dg_add_node(planner->decision_graph);
            if (node_id != -1) {
                planner->map_nodes[node_id].lane_id = i;
                planner->map_nodes[node_id].progress = p;
                
                int idx_in_lane = planner->lane_id_to_node_ids_count[i];
                if (idx_in_lane < MAX_NODES_PER_LANE) {
                    planner->lane_id_to_node_ids_list[i][idx_in_lane] = node_id;
                    planner->lane_id_to_node_ids_count[i]++;
                }
            }
        }
        #undef ADD_POINT
    }

    // 2. Generate Edges
    int num_nodes = planner->decision_graph->num_nodes;
    for (int u_id = 0; u_id < num_nodes; ++u_id) {
        MapNode* u = &planner->map_nodes[u_id];
        Lane* lane_u = map_get_lane(planner->map, u->lane_id);

        // Highway avoidance penalty
        double highway_penalty = 1.0;
        if (planner->avoid_highways && lane_u->road_id >= 0) {
            Road* road = map_get_road(planner->map, lane_u->road_id);
            if (road && road->num_lanes >= 3) {
                highway_penalty = HIGHWAY_AVOIDANCE_PENALTY;
            }
        }

        // A. Longitudinal (Next node on same lane)
        int* lane_nodes = planner->lane_id_to_node_ids_list[u->lane_id];
        int count = planner->lane_id_to_node_ids_count[u->lane_id];
        
        for(int k=0; k<count-1; ++k) {
            if (lane_nodes[k] == u_id) {
                int v_id = lane_nodes[k+1];
                MapNode* v = &planner->map_nodes[v_id];
                double dist = v->progress - u->progress;
                if (dist > 0) {
                    double traffic_flow = planner->traffic_flow_per_edge[u->lane_id][k];
                    double cost = get_cost_with_traffic(dist, lane_u->speed_limit, planner->consider_speed_limits, planner->use_live_traffic_info, traffic_flow);
                    dg_add_edge_w(planner->decision_graph, u_id, v_id, cost * highway_penalty);
                }
                break;
            }
        }

        // B. Connections (End of Lane -> Start of Next)
        if (fabs(u->progress - lane_u->length) < 0.1) {
            for (int k = 0; k < 3; ++k) {
                LaneId next_id = lane_u->connections[k];
                if (next_id != ID_NULL) {
                    // Find start node of next_id (first node with that lane_id)
                    if (planner->lane_id_to_node_ids_count[next_id] > 0) {
                        int v_id = planner->lane_id_to_node_ids_list[next_id][0];
                        if (planner->map_nodes[v_id].progress < 0.1) {
                            dg_add_edge_w(planner->decision_graph, u_id, v_id, 0.0);
                        }
                    }
                }
            }
        }

        // C. Adjacents (Lane Change)
        for (int k = 0; k < 2; ++k) {
            if (u->progress < LANE_CHANGE_INTERVAL) continue; // Don't allow lane changes at the very start of the lane. Our NPCs do not do that.
            if (lane_u->length - u->progress < LANE_CHANGE_INTERVAL) continue; // Don't allow lane changes at the very end of the lane, either, as they are equivalent to lane changes at the very start of the next lane.
            LaneId adj_id = lane_u->adjacents[k];
            if (adj_id != ID_NULL) {
                if (adj_id == lane_u->merges_into_id || adj_id == lane_u->exit_lane_id) continue;

                Lane* adj = map_get_lane(planner->map, adj_id);
                if (adj->direction != lane_u->direction) continue;

                // Find closest node on adj_id
                int best_v = -1;
                double min_diff = DBL_MAX;
                
                int* adj_nodes = planner->lane_id_to_node_ids_list[adj_id];
                int adj_count = planner->lane_id_to_node_ids_count[adj_id];

                for(int j=0; j<adj_count; ++j) {
                    int v_id = adj_nodes[j];
                    double diff = fabs(planner->map_nodes[v_id].progress - u->progress);
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_v = v_id;
                    }
                }

                if (best_v != -1 && min_diff < 5.0) {
                    dg_add_edge_w(planner->decision_graph, u_id, best_v, 1.0); // Penalty (1 second if considering speed limits else 1 meter)
                }
            }
        }

        // D. Merge
        if (lane_u->merges_into_id != ID_NULL) {
            LaneId target_id = lane_u->merges_into_id;
            Lane* target = map_get_lane(planner->map, target_id);
            
            double next_entry = u->progress + lane_u->merges_into_start * target->length;
            double max_entry = lane_u->merges_into_end * target->length;

            if (next_entry <= max_entry + 0.1) {
                 int best_v = -1;
                 double min_diff = DBL_MAX;
                 
                 int* target_nodes = planner->lane_id_to_node_ids_list[target_id];
                 int target_count = planner->lane_id_to_node_ids_count[target_id];

                 for(int j=0; j<target_count; ++j) {
                    int v_id = target_nodes[j];
                    double diff = fabs(planner->map_nodes[v_id].progress - next_entry);
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_v = v_id;
                    }
                 }
                 if (best_v != -1 && min_diff < 2.0) {
                     dg_add_edge_w(planner->decision_graph, u_id, best_v, 1.0); // Penalty (1 second if considering speed limits else 1 meter)
                 }
            }
        }

        // E. Exit
        if (lane_u->exit_lane_id != ID_NULL) {
            LaneId target_id = lane_u->exit_lane_id;
            
            double exit_start = lane_u->exit_lane_start * lane_u->length;
            double exit_end = lane_u->exit_lane_end * lane_u->length;

            if (u->progress >= exit_start && u->progress <= exit_end) {
                double target_proj = u->progress - exit_start;
                
                int best_v = -1;
                 double min_diff = DBL_MAX;
                 
                 int* target_nodes = planner->lane_id_to_node_ids_list[target_id];
                 int target_count = planner->lane_id_to_node_ids_count[target_id];

                 for(int j=0; j<target_count; ++j) {
                    int v_id = target_nodes[j];
                    double diff = fabs(planner->map_nodes[v_id].progress - target_proj);
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_v = v_id;
                    }
                 }
                 if (best_v != -1 && min_diff < 2.0) {
                     dg_add_edge_w(planner->decision_graph, u_id, best_v, 1.0); // Penalty (1 second if considering speed limits else 1 meter)
                 }
            }
        }
    }
}

PathPlanner* path_planner_create(Map* map, bool consider_speed_limits) {
    PathPlanner* planner = (PathPlanner*)malloc(sizeof(PathPlanner));
    if (planner) {
        planner->map = map;
        planner->consider_speed_limits = consider_speed_limits;
        planner->use_live_traffic_info = false;
        planner->avoid_highways = false;
        memset(planner->traffic_flow_per_edge, 0, sizeof(planner->traffic_flow_per_edge));
        planner->last_traffic_update_time = -1.0;
        planner->start_lane_id = ID_NULL;
        planner->end_lane_id = ID_NULL;
        planner->path_exists = false;
        memset(planner->solution_nodes, 0, sizeof(planner->solution_nodes));
        memset(planner->solution_actions, 0, sizeof(planner->solution_actions));
        planner->num_solution_actions = 0;
        planner->optimal_cost = 0.0;
        planner->path_exists = false;
        
        planner->decision_graph = dg_create(MAX_GRAPH_NODES);
        memset(planner->map_nodes, 0, sizeof(planner->map_nodes));
        memset(planner->solution_intermediate_node_ids, 0, sizeof(planner->solution_intermediate_node_ids));
        memset(planner->lane_id_to_node_ids_count, 0, sizeof(planner->lane_id_to_node_ids_count));
        for (int i = 0; i < MAX_NUM_LANES; ++i) {
            for (int j = 0; j < MAX_NODES_PER_LANE; ++j) {
                planner->lane_id_to_node_ids_list[i][j] = -1;
            }
        }

        path_planner_build_graph(planner);
    }
    return planner;
}

void path_planner_free(PathPlanner* planner) {
    if (planner) {
        if (planner->decision_graph) {
            dg_free(planner->decision_graph);
        }
        free(planner);
    }
}

static double get_cost_with_traffic(double distance, double speed_limit, bool consider_speed_limits, bool use_live_traffic, double traffic_flow) {
    if (distance < 0) distance = 0;
    if (consider_speed_limits) {
        if (use_live_traffic && traffic_flow > 0.0) {
            // Use live traffic flow speed, but clamp to a minimum
            double effective_speed = traffic_flow;
            if (effective_speed < TRAFFIC_FLOW_MIN_SPEED) effective_speed = TRAFFIC_FLOW_MIN_SPEED;
            return distance / effective_speed;
        }
        if (speed_limit < 1.0) speed_limit = 1.0;
        return distance / speed_limit;
    }
    return distance;
}

// Compute the cost of traversing a sub-range [from_progress, to_progress] on a lane,
// summing per-edge traffic flow costs across all overlapping edge segments.
static double get_lane_segment_cost(PathPlanner* planner, LaneId lane_id, Meters from_progress, Meters to_progress) {
    Lane* lane = map_get_lane(planner->map, lane_id);
    if (!lane || to_progress <= from_progress) {
        return get_cost_with_traffic(to_progress - from_progress, lane ? lane->speed_limit : 1.0,
                                     planner->consider_speed_limits, false, 0.0);
    }

    if (!planner->use_live_traffic_info) {
        return get_cost_with_traffic(to_progress - from_progress, lane->speed_limit,
                                     planner->consider_speed_limits, false, 0.0);
    }

    // Sum costs across edge segments that overlap [from_progress, to_progress]
    int node_count = planner->lane_id_to_node_ids_count[lane_id];
    double total_cost = 0.0;
    bool found_any = false;

    for (int k = 0; k < node_count - 1; k++) {
        int u_id = planner->lane_id_to_node_ids_list[lane_id][k];
        int v_id = planner->lane_id_to_node_ids_list[lane_id][k + 1];
        double seg_start = planner->map_nodes[u_id].progress;
        double seg_end   = planner->map_nodes[v_id].progress;

        double overlap_start = seg_start > from_progress ? seg_start : from_progress;
        double overlap_end   = seg_end   < to_progress   ? seg_end   : to_progress;

        if (overlap_start < overlap_end) {
            double dist = overlap_end - overlap_start;
            double flow = planner->traffic_flow_per_edge[lane_id][k];
            total_cost += get_cost_with_traffic(dist, lane->speed_limit,
                                                planner->consider_speed_limits,
                                                planner->use_live_traffic_info, flow);
            found_any = true;
        }
    }

    if (!found_any) {
        // Fallback: no edge segments cover this range, use speed limit
        return get_cost_with_traffic(to_progress - from_progress, lane->speed_limit,
                                     planner->consider_speed_limits, false, 0.0);
    }
    return total_cost;
}


static NavAction infer_action(Map* map, MapNode u, MapNode v, bool* trivial_out) {
    *trivial_out = true;
    if (u.lane_id == v.lane_id) return NAV_STRAIGHT;

    Lane* lane_u = map_get_lane(map, u.lane_id);
    if (!lane_u) return NAV_NONE;

    Lane* lane_v = map_get_lane(map, v.lane_id);
    if (!lane_v) return NAV_NONE;

    // Simple lane continuation:
    if (lane_get_num_connections(lane_u) == 1 && lane_u->connections[INDICATOR_NONE] == v.lane_id && lane_get_num_connections_incoming(lane_v) == 1) {
        return NAV_STRAIGHT;
    }

    // If we are already *on* an intersection and turning into a road, consider it continuing straight
    if (lane_u->is_at_intersection && lane_get_num_connections(lane_u) == 1) {
        return NAV_STRAIGHT;
    }

    *trivial_out = false;   // all below are non-trivial and useful for navigation

    // Connections
    for (int k = 0; k < 3; ++k) {
        if (lane_u->connections[k] == v.lane_id) {
            if (k == 0) return NAV_TURN_LEFT;
            if (k == 1) return NAV_KEEP_STRAIGHT ;
            if (k == 2) return NAV_TURN_RIGHT;
        }
    }

    // Adjacents
    if (lane_u->adjacents[0] == v.lane_id) return NAV_LANE_CHANGE_LEFT;
    if (lane_u->adjacents[1] == v.lane_id) return NAV_LANE_CHANGE_RIGHT;

    // Merge
    if (lane_u->merges_into_id == v.lane_id) {
        if (lane_u->adjacents[1] == v.lane_id) return NAV_MERGE_RIGHT;
        return NAV_MERGE_LEFT;
    }

    // Exit
    if (lane_u->exit_lane_id == v.lane_id) {
        if (lane_u->adjacents[0] == v.lane_id) return NAV_EXIT_LEFT;
        return NAV_EXIT_RIGHT;
    }

    *trivial_out = true;
    return NAV_NONE;
}

void path_planner_compute_shortest_path(PathPlanner* planner, const Lane* start_lane, Meters start_progress, const Lane* end_lane, Meters end_progress) {
    if (!planner || !start_lane || !end_lane || !planner->map) return;

    planner->start_lane_id = start_lane->id;
    planner->start_progress = start_progress;
    planner->end_lane_id = end_lane->id;
    planner->end_progress = end_progress;

    // when there is no path:
    planner->path_exists = false;
    planner->optimal_cost = DBL_MAX;
    planner->num_solution_actions = 0;
    planner->num_solution_actions_non_trivial = 0;

    // if we are on the same lane and start_progress <= end_progress, trivial path
    if (start_lane->id == end_lane->id && start_progress <= end_progress) {
        planner->path_exists = true;
        planner->optimal_cost = get_lane_segment_cost(planner, start_lane->id, start_progress, end_progress);
        planner->num_solution_actions = 1;
        planner->num_solution_actions_non_trivial = 1;
        planner->solution_nodes[0].lane_id = start_lane->id;
        planner->solution_nodes[0].progress = start_progress;
        planner->solution_nodes[1].lane_id = end_lane->id;
        planner->solution_nodes[1].progress = end_progress;
        planner->solution_actions[0] = NAV_STRAIGHT;
        planner->solution_actions_non_trivial[0] = NAV_STRAIGHT;
        return;
    }

    // 1. Find Node_1 (Closest node on start_lane just after start_progress)
    int node1_idx = -1;
    double node1_closest_dist = DBL_MAX;
    
    // TODO later: can optimize this if our lane_id_to_node_ids_list is constructed sorted
    for (int i = 0; i < planner->lane_id_to_node_ids_count[start_lane->id]; ++i) {
        int node_idx = planner->lane_id_to_node_ids_list[start_lane->id][i];
        if (planner->map_nodes[node_idx].progress >= start_progress) {
            double dist = planner->map_nodes[node_idx].progress - start_progress;
            if (dist < node1_closest_dist) {
                node1_closest_dist = dist;
                node1_idx = node_idx;
            }
        }
    }

    // 2. Find Node_end_minus_1 (Closest node on end_lane just before end_progress)
    int node_end_minus_1_idx = -1;
    double node_end_minus_1_closest_dist = DBL_MAX;
    for (int i = 0; i < planner->lane_id_to_node_ids_count[end_lane->id]; ++i) {
        int node_idx = planner->lane_id_to_node_ids_list[end_lane->id][i];
        if (planner->map_nodes[node_idx].progress <= end_progress) {
            double dist = end_progress - planner->map_nodes[node_idx].progress;
            if (dist < node_end_minus_1_closest_dist) {
                node_end_minus_1_closest_dist = dist;
                node_end_minus_1_idx = node_idx;
            }
        }
    }

    if (node1_idx == -1 || node_end_minus_1_idx == -1) return;

    // Compute fractional segment costs in the same units as the graph edge costs
    // (time when consider_speed_limits is on, meters otherwise). Using raw distances
    // here would cause unit mismatch and wild ETA fluctuations in time-cost mode.
    double node1_cost = get_lane_segment_cost(planner, start_lane->id, start_progress,
                                              planner->map_nodes[node1_idx].progress);
    double node_end_cost = get_lane_segment_cost(planner, end_lane->id,
                                                 planner->map_nodes[node_end_minus_1_idx].progress,
                                                 end_progress);

    // 3. Call Dijkstra/A*
    double out_cost = 0.0;
    int path_num_nodes = dg_astar_path(planner->decision_graph, node1_idx, node_end_minus_1_idx, astar_heuristic, planner, planner->solution_intermediate_node_ids, MAX_SOLUTION_CAPACITY - 2, &out_cost);
    
    if (path_num_nodes > 0) {
        planner->path_exists = true;
        planner->optimal_cost = out_cost + node1_cost + node_end_cost;
        planner->num_solution_actions = path_num_nodes + 2 - 1; // +2 for start and end, -1 for actions
        // Fill solution nodes
        // Start
        planner->solution_nodes[0].lane_id = start_lane->id;
        planner->solution_nodes[0].progress = start_progress;
        planner->solution_actions[0] = NAV_STRAIGHT; // from start to first node
        // Intermediate
        planner->num_solution_actions_non_trivial = 0;
        for (int i = 0; i < path_num_nodes; ++i) {
            int node_idx = planner->solution_intermediate_node_ids[i];
            planner->solution_nodes[i + 1] = planner->map_nodes[node_idx];
            
            if (i > 0) {
                bool trivial = false;
                planner->solution_actions[i] = infer_action(planner->map, planner->solution_nodes[i], planner->solution_nodes[i+1], &trivial);
                if (!trivial) {
                    planner->solution_actions_non_trivial[planner->num_solution_actions_non_trivial++] = planner->solution_actions[i];
                }
            }
        }
        // End
        planner->solution_nodes[path_num_nodes + 1].lane_id = end_lane->id;
        planner->solution_nodes[path_num_nodes + 1].progress = end_progress;
        planner->solution_actions[path_num_nodes] = NAV_STRAIGHT;
        planner->solution_actions_non_trivial[planner->num_solution_actions_non_trivial++] = NAV_STRAIGHT;
    } else if (path_num_nodes == 0) {
        // no path
        LOG_ERROR("Path Planner: No path found. Start Lane ID: %d, Start Progress: %.2f, End Lane ID: %d, End Progress: %.2f.", 
              start_lane->id, start_progress, end_lane->id, end_progress);
    } else {
        // error
        LOG_ERROR("Path Planner Dijkstra error: %d. Start Lane ID: %d, Start Progress: %.2f, End Lane ID: %d, End Progress: %.2f.", 
              path_num_nodes, start_lane->id, start_progress, end_lane->id, end_progress);
        exit(EXIT_FAILURE);
    }
}



NavAction path_planner_get_solution_action(const PathPlanner* planner, bool ignore_trivial, int action_index) {
    if (!planner || !planner->path_exists || planner->num_solution_actions == 0) {
        return NAV_NONE;
    }
    if (ignore_trivial) {
        if (action_index < 0 || action_index >= planner->num_solution_actions_non_trivial) {
            return NAV_NONE;
        }
        return planner->solution_actions_non_trivial[action_index];
    } else {
        if (action_index < 0 || action_index >= planner->num_solution_actions) {
            return NAV_NONE;
        }
        return planner->solution_actions[action_index];
    }
}


// ---------------------------------------------------------------------------
// Live Traffic Flow
// ---------------------------------------------------------------------------

static void path_planner_rebuild_graph(PathPlanner* planner) {
    // Free old graph and create a fresh one, then rebuild all nodes and edges
    if (planner->decision_graph) {
        dg_free(planner->decision_graph);
    }
    planner->decision_graph = dg_create(MAX_GRAPH_NODES);
    memset(planner->map_nodes, 0, sizeof(planner->map_nodes));
    memset(planner->lane_id_to_node_ids_count, 0, sizeof(planner->lane_id_to_node_ids_count));
    for (int i = 0; i < MAX_NUM_LANES; ++i) {
        for (int j = 0; j < MAX_NODES_PER_LANE; ++j) {
            planner->lane_id_to_node_ids_list[i][j] = -1;
        }
    }
    path_planner_build_graph(planner);
}

void path_planner_update_traffic_flow(Simulation* sim, PathPlanner* planner) {
    if (!sim || !planner || !planner->map) return;

    Map* map = &sim->map;

    // Compute actual elapsed sim-time since last call
    double elapsed;
    if (planner->last_traffic_update_time < 0.0) {
        elapsed = 0.0; // first call — will initialize directly
    } else {
        elapsed = sim->time - planner->last_traffic_update_time;
    }
    planner->last_traffic_update_time = sim->time;

    if (elapsed < 0.0) elapsed = 0.0;

    // EMA alpha from elapsed time: alpha = 1 - exp(-elapsed / tau)
    // This is frame-rate independent.
     double alpha = (elapsed > 0.0) ? (1.0 - exp(-elapsed / TRAFFIC_FLOW_TAU)) : 0.0;

    // Update traffic flow per edge segment: for each lane, for each consecutive
    // pair of graph nodes (which defines an edge), bucket the cars by their
    // progress into the correct segment and compute average speed.
    for (int lane_idx = 0; lane_idx < map->num_lanes; ++lane_idx) {
        Lane* lane = map_get_lane(map, lane_idx);
        if (!lane) continue;

        int node_count = planner->lane_id_to_node_ids_count[lane_idx];
        if (node_count < 2) continue; // no edge segments on this lane

        int num_cars = lane->num_cars;
        // if (num_cars == 0) continue; // no cars — keep previous estimates

        for (int k = 0; k < node_count - 1; k++) {
            int u_id = planner->lane_id_to_node_ids_list[lane_idx][k];
            int v_id = planner->lane_id_to_node_ids_list[lane_idx][k + 1];
            double seg_start = planner->map_nodes[u_id].progress;
            double seg_end   = planner->map_nodes[v_id].progress;

            if (planner->traffic_flow_per_edge[lane_idx][k] <= 0.0) {
                // First observation or first call — initialize with speed limit to avoid zero-cost edges
                planner->traffic_flow_per_edge[lane_idx][k] = lane->speed_limit;
            }

            // Accumulate speeds of cars within this segment
            double total_speed = 0.0;
            int car_count = 0;

            for (int c = 0; c < num_cars; ++c) {
                Car* car = sim_get_car(sim, lane->cars_ids[c]);
                if (!car) continue;
                Meters car_prog = car_get_lane_progress_meters(car);
                // Car belongs to this segment if progress is in [seg_start, seg_end)
                // (last segment uses inclusive end to capture cars at the very end)
                bool in_segment = (k == node_count - 2)
                    ? (car_prog >= seg_start && car_prog <= seg_end)
                    : (car_prog >= seg_start && car_prog <  seg_end);
                if (in_segment) {
                    total_speed += car_get_speed(car);
                    car_count++;
                }
            }
            
            double avg_speed;
            if (car_count > 0) {
                avg_speed = total_speed / (double)car_count;
            } else {
                avg_speed = lane->speed_limit; // no cars means free flow at speed limit
            }

            // Exponential moving average update
            planner->traffic_flow_per_edge[lane_idx][k] = (1.0 - alpha) * planner->traffic_flow_per_edge[lane_idx][k] + alpha * avg_speed;
        }
    }

    // Rebuild the decision graph so edge weights reflect updated traffic flow
    if (planner->use_live_traffic_info) {
        path_planner_rebuild_graph(planner);
    }
}

void path_planner_reset_traffic_stats(PathPlanner* planner) {
    if (planner) {
        // Reset the traffic flow stats array to zero
        memset(planner->traffic_flow_per_edge, 0, sizeof(planner->traffic_flow_per_edge));
        planner->last_traffic_update_time = -1.0;
        
        // If live traffic info is in use, rebuild the graph to clear out old weights
        if (planner->use_live_traffic_info) {
             path_planner_rebuild_graph(planner);
        }
    }
}
