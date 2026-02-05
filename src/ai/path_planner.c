#include "ai.h"
#include "map.h"
#include "utils.h"
#include "logging.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

// Constants for graph generation
#define LANE_CHANGE_INTERVAL 40.0 // Meters between lane change nodes


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


static double get_cost(double distance, double speed_limit, bool consider_speed_limits);

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

        // A. Longitudinal (Next node on same lane)
        int* lane_nodes = planner->lane_id_to_node_ids_list[u->lane_id];
        int count = planner->lane_id_to_node_ids_count[u->lane_id];
        
        for(int k=0; k<count-1; ++k) {
            if (lane_nodes[k] == u_id) {
                int v_id = lane_nodes[k+1];
                MapNode* v = &planner->map_nodes[v_id];
                double dist = v->progress - u->progress;
                if (dist > 0) {
                    dg_add_edge_w(planner->decision_graph, u_id, v_id, get_cost(dist, lane_u->speed_limit, planner->consider_speed_limits));
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

static double get_cost(double distance, double speed_limit, bool consider_speed_limits) {
    if (distance < 0) distance = 0;
    if (consider_speed_limits) {
        if (speed_limit < 1.0) speed_limit = 1.0;
        return distance / speed_limit;
    }
    return distance;
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
        planner->optimal_cost = get_cost(end_progress - start_progress, start_lane->speed_limit, planner->consider_speed_limits);
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

    // 3. Call Dijkstra -> A*
    double out_cost = 0.0;
    int path_num_nodes = dg_astar_path(planner->decision_graph, node1_idx, node_end_minus_1_idx, astar_heuristic, planner, planner->solution_intermediate_node_ids, MAX_SOLUTION_CAPACITY - 2, &out_cost);
    
    if (path_num_nodes > 0) {
        planner->path_exists = true;
        planner->optimal_cost = out_cost + node1_closest_dist + node_end_minus_1_closest_dist;
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