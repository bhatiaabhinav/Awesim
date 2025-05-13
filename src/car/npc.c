#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "car.h"

void NPC_car_make_decisions(Car* self, const Simulation* sim) {

    // Some current variables:
    const Lane* lane = self->lane;
    Meters lane_progress_m = car_get_lane_progress_meters(self);
    Meters lane_end_distance_m = lane->length - lane_progress_m;
    MetersPerSecond speed = car_get_speed(self);
    MetersPerSecondSquared accel = car_get_acceleration(self);
    double pid_k_preferred = acceleration_mode_pid_k(self->preferences.acceleration_mode);
    double pid_k_hard = acceleration_mode_pid_k(SPORT_MODE);
    double pid_k_emergency = acceleration_mode_pid_k(EMERGENCY_MODE);

    // Some useful variables:
    double pid_k_required = pid_min_k(lane_end_distance_m - 2.0, -speed);     // minimum k required to stop without overshooting. using formula k = (speed_error / distance_error)^2.
    double panic_ratio = pid_k_required / pid_k_preferred;  // ratio of required k to preferred k. Start panicking if this ratio is > 1.0. 
    bool is_comfy_stopping_impossible = panic_ratio > 1.0;
    bool is_hard_stopping_impossible = pid_k_required > pid_k_hard;
    bool is_emergency_stopping_impossible = pid_k_required > pid_k_emergency;
    bool close_to_end_of_lane = lane_end_distance_m < 5.0;
    

    // NPC Variables for decision-making:
    NPCState state = self->npc_state;
    state.approach_mode = (state.approach_mode ? panic_ratio > 0.8 : panic_ratio >= 0.9) || close_to_end_of_lane;  // enter approach mode if panic ratio is > 0.9 and exit if it is < 0.8. Or if close to end of lane.
    DirectionIntent turn_intent;
    DirectionIntent lane_change_intent;
    const Lane* lane_change_target;


    // ------------- Decide turn and lane change intent ----------------
    if (state.approach_mode) {
        turn_intent = car_get_turn_intent(self);    // keep the turn intent   
        if (!is_turn_possible(lane, turn_intent)) {               // a leftover from prev lane may not be valid now.
            // printf("Turn intent %d not possible. Trying to correct it\n", turn_intent);
            turn_intent = get_legitimate_turn_intent_closest_to(lane, rand_int_range(0, 2));  // note: can still be INTENT_NA if it is a dead end.
            // printf("Turn intent corrected to %d\n", turn_intent);
        }
        lane_change_intent = INTENT_STRAIGHT;       // no lane change
        lane_change_target = lane;                   // no lane change
    }
    else {
        turn_intent = get_legitimate_turn_intent_closest_to(lane, rand_int_range(0, 2));
        double r = rand_0_to_1();
        if (r <= 0.9) {
            lane_change_intent = INTENT_STRAIGHT;                       // 90% chance to not change lane
            lane_change_target = lane;
        } else if (r <= 0.95 && is_left_lane_change_possible(lane)) {
            lane_change_intent = INTENT_LEFT;                           // 5% chance to change to left lane
            lane_change_target = lane->adjacent_left;
        } else if (r <= 1.0 && is_right_lane_change_possible(lane)) {
            lane_change_intent = INTENT_RIGHT;                          // 5% chance to change to right lane
            lane_change_target = lane->adjacent_right;
        } else {
            lane_change_intent = INTENT_STRAIGHT;                       // no lane change
            lane_change_target = lane;
        }
    }
    // handle dead ends:
    bool is_dead_end = lane_num_connections(lane) == 0;
    if (is_dead_end) {
        if (turn_intent != INTENT_NA) {
            fprintf(stderr, "ERROR: Turn intent should be NA for dead ends. How did this happen?\n");
            exit(1);
        }
        if (lane_check_merge_available(lane) || is_left_lane_change_possible(lane)) {
            lane_change_intent = INTENT_LEFT;       // try to merge left
            lane_change_target = lane->merges_into;
            // printf("Taking actions towards merging\n");
        } else {
            fprintf(stderr, "WARN: Dead end with no merge available!\n");
            lane_change_intent = INTENT_STRAIGHT;   // no lane change
            lane_change_target = lane;
        }
    }
    // cancel lane change if it would lead to a collision:
    if (lane_change_target != lane && would_collide_on_lane_change(self, lane_change_target, 8.0)) {
        // TODO: should try to create some gap to be able to change lanes.
        // for now, just cancel the lane change.
        lane_change_intent = INTENT_STRAIGHT;
        lane_change_target = lane;
    }
    // ------------------------------------------------------------------


    // ----------- Decide speed error and distance error -----------------
    if (state.approach_mode) {
        const Lane* turning_to_lane = turn_intent_leads_to_lane(lane, turn_intent); // could be NULL if this lane is a dead end (corresponding to INTENT_NA)
        GoSituationWithCountdown go_situation = go_situation_compute(lane, turn_intent);
        // printf("Turning from lane %p to lane %p due to turn intent %d, with light state %d\n", lane, turning_to_lane, turn_intent, go_situation.state);
        switch(go_situation.state) {
        case GO_GREEN:
            // printf("Green light\n");
            state.braking = false;
            break;
        case GO_YELLOW:
            // printf("Yellow light\n");
            if (is_hard_stopping_impossible) {
                state.braking = false;  // Too late to brake — run the yellow
            } else {
                if (is_comfy_stopping_impossible) {
                    // you have the option to run the yellow or to break hard
                    state.braking = !self->preferences.run_yellow_light;  // run the yellow if the user prefers it
                } else {
                    state.braking = true;  // comfy brake
                }
            }
            break;
        case GO_RED:
            // printf("Red light\n");
            if (is_emergency_stopping_impossible) {
                state.braking = false;  // Too late to brake — run the red
            } else {
                state.braking = true;
            }
            break;
        case GO_STOP:
            // printf("4-way stop\n");
            if (is_emergency_stopping_impossible) {
                state.braking = false;  // Too late to brake — run the red
            } else {
                state.braking = true;
            }
            break;
        case GO_ILLEGAL:
            // fprintf(stderr, "WARN: Likely approaching a dead end! Turn intent was %d\n", turn_intent);
            state.braking = true;
            break;
        default:
            fprintf(stderr, "ERROR: Unknown light state\n");
            exit(1);
            break;
        }
        if (state.braking) {
            state.speed_error = -speed; // stop
            state.distance_error = lane_end_distance_m - 2.0; // stop before the end of the lane
        } else {
            if(turning_to_lane == NULL) {
                fprintf(stderr, "ERROR: Turning to lane is NULL. Turn intent was %d and light state was %d\n", turn_intent, go_situation.state);
                exit(1);
            }
            state.speed_error = turning_to_lane->speed_limit - speed; // speed limit of the next lane
            state.speed_error += self->preferences.average_speed_offset; // user preference
            state.distance_error = 0;
        }
        state.cruising = false;
    } else {
        state.speed_error = lane->speed_limit - speed;
        state.speed_error += self->preferences.average_speed_offset; // user preference
        state.distance_error = 0;
        state.braking = false;
        state.cruising = true;
    }
    // ---------------------------------------------------------------------------------

    // ----------------------------- Set acceleration ---------------------------------
    // decide acceleration based on speed and distance errors:
    double pid_k = pid_k_preferred;
    if (state.braking) {
        pid_k = fmax(pid_k_preferred, pid_min_k(state.distance_error, state.speed_error));
        pid_k = fmin(pid_k, pid_k_emergency);
        if (pid_k == pid_k_emergency) {
            printf("Emergency braking (k = %.6f). Distance error: %.6f. Speed error: %.6f\n", pid_k, state.distance_error, state.speed_error);
        }
        if (state.pid_k_braking && pid_k > state.pid_k_braking + 0.01) {
            fprintf(stderr, "PID k increased from %.6f to %.6f in a single braking session. This should not happen.\n", state.pid_k_braking, pid_k);
            exit(1);
        }
        state.pid_k_braking = pid_k;
    } else {
        state.pid_k_braking = 0;   // mark as not braking
    }
    accel = pid_acceleration(state.distance_error, state.speed_error, pid_k);
    // ---------------------------------------------------------------------------------


    // ---------------- Set car decision variables for the simulator to process -----------------
    self->npc_state = state; // update the NPC state
    car_set_turn_intent(self, turn_intent, false);
    car_set_lane_change_intent(self, lane_change_intent);
    car_set_acceleration(self, accel);
}



//  ------ Decision-making helper functions ----

bool would_collide_on_lane_change(const Car* self, const Lane* target_lane, const Meters buffer_m) {
    if (target_lane == NULL) {
        return false;
    }
    double my_progress = car_get_lane_progress(self);
    double my_hypothetical_target_progress = (target_lane == self->lane->merges_into) ? (self->lane->merges_into_start + my_progress) : my_progress;
    Meters my_hypothetical_target_progress_m = target_lane->length * my_hypothetical_target_progress;
    for (int i = 0; i < target_lane->num_cars; i++) {
        const Car* other_car = target_lane->cars[i];
        if (other_car == self) {
            continue; // skip self
        }
        Meters other_car_progress_m = car_get_lane_progress_meters(other_car);
        if (fabs(my_hypothetical_target_progress_m - other_car_progress_m) < car_get_length(self) + buffer_m) {
            return true;
        }
    }
    return false;
}

bool is_left_lane_change_possible(const Lane* lane) {
    return (lane->adjacent_left != NULL && lane->adjacent_left->direction == lane->direction);
}

bool is_right_lane_change_possible(const Lane* lane) {
    if (lane->adjacent_right != NULL) {
        // assert(lane->adjacent_right->direction == lane->direction && "Right lane has a different direction. Map error!");
    }
    return (lane->adjacent_right != NULL && lane->adjacent_right->direction == lane->direction);
}

MetersPerSecondSquared pid_acceleration(const Meters distance_error, const MetersPerSecond speed_error, const double pid_k) {
    const double k = pid_k;
    double a = 2 * sqrt(k) * speed_error + k * distance_error;
    return a;
}

double pid_min_k(const Meters distance_error, const MetersPerSecond speed_error) {
    if (fabs(speed_error) < 1e-4 && fabs(distance_error) < 1e-4) {
        return 0.0; // No need for acceleration if both errors are negligible
    }
    double ratio = speed_error / (distance_error + 1e-6);
    return ratio * ratio;
}

DirectionIntent turn_intent_given_next_lane(const Lane* lane_from, const Lane* lane_to) {
    if (lane_from->adjacent_left == lane_to) {
        return INTENT_LEFT;
    } else if (lane_from->adjacent_right == lane_to) {
        return INTENT_RIGHT;
    } else if (lane_from->for_straight_connects_to == lane_to) {
        return INTENT_STRAIGHT;
    } else {
        if (lane_to == NULL) {
            return INTENT_NA;
        } else {
            fprintf(stderr, "Lane_to is not connected to lane_from\n");
            exit(1);
        }
    }
}

bool is_turn_possible(const Lane* lane_from, DirectionIntent turn_intent) {
    if (turn_intent == INTENT_LEFT) {
        return lane_from->for_left_connects_to != NULL;
    } else if (turn_intent == INTENT_RIGHT) {
        return lane_from->for_right_connects_to != NULL;
    } else if (turn_intent == INTENT_STRAIGHT) {
        return lane_from->for_straight_connects_to != NULL;
    } else {
        return false; // Invalid turn intent
    }
}

DirectionIntent get_legitimate_turn_intent_closest_to(const Lane* lane_from, DirectionIntent turn_intent) {
    if (is_turn_possible(lane_from, turn_intent)) {
        return turn_intent;
    }

    // Check for the closest possible turn intent
    if (turn_intent == INTENT_LEFT) {
        if (is_turn_possible(lane_from, INTENT_STRAIGHT)) {
            return INTENT_STRAIGHT;
        } else if (is_turn_possible(lane_from, INTENT_RIGHT)) {
            return INTENT_RIGHT;
        }
    } else if (turn_intent == INTENT_RIGHT) {
        if (is_turn_possible(lane_from, INTENT_STRAIGHT)) {
            return INTENT_STRAIGHT;
        } else if (is_turn_possible(lane_from, INTENT_LEFT)) {
            return INTENT_LEFT;
        }
    } else if (turn_intent == INTENT_STRAIGHT) {
        if (is_turn_possible(lane_from, INTENT_RIGHT)) {
            return INTENT_RIGHT;
        } else if (is_turn_possible(lane_from, INTENT_LEFT)) {
            return INTENT_LEFT;
        }
    }

    return INTENT_NA; // No valid turn intent found
}

const Lane* turn_intent_leads_to_lane(const Lane* lane, DirectionIntent turn_intent) {
    switch (turn_intent) {
        case INTENT_LEFT:
            return lane->for_left_connects_to;
        case INTENT_RIGHT:
            return lane->for_right_connects_to;
        case INTENT_STRAIGHT:
            return lane->for_straight_connects_to;
        default:
            return NULL; // Invalid turn intent
    }
}


GoSituationWithCountdown go_situation_compute(const Lane* from_lane, DirectionIntent turn_intent) {
    if (!is_turn_possible(from_lane, turn_intent)) {
        // fprintf(stderr, "Turn intent not possible. Returning illegal go state.\n");
        return (GoSituationWithCountdown){GO_ILLEGAL, 1e9};
    }
    const Intersection* intersection = road_leads_to_intersection(from_lane->road);
    if (!intersection) {
        return (GoSituationWithCountdown){GO_GREEN, 1e9}; // no intersection, so green light
    }
    // printf("Going into intersection\n");
    const bool free_right = true;
    Direction lane_dir = from_lane->direction;
    const bool is_NS = (lane_dir == DIRECTION_NORTH || lane_dir == DIRECTION_SOUTH);
    IntersectionState intersection_state = intersection->state;
    GoSituation go_situation;
    if (intersection_state == FOUR_WAY_STOP) {
        return (GoSituationWithCountdown){GO_STOP, 1e9};
    }
    if (turn_intent == INTENT_RIGHT && free_right) {
        return (GoSituationWithCountdown){GO_GREEN, 1e9};         // right turn is always allowed
    }

    switch (intersection_state) {
        case NS_GREEN_EW_RED:
            if (is_NS) {
                go_situation = GO_GREEN;
            } else {
                go_situation = GO_RED;
            }
            break;
        case NS_YELLOW_EW_RED:
            if (is_NS) {
                go_situation = GO_YELLOW;
            } else {
                go_situation = GO_RED;
            }
            break;
        case ALL_RED_BEFORE_EW_GREEN:
            go_situation = GO_RED;
            break;
        case NS_RED_EW_GREEN:
            if (is_NS) {
                go_situation = GO_RED;
            } else {
                go_situation = GO_GREEN;
            }
            break;
        case EW_YELLOW_NS_RED:
            if (is_NS) {
                go_situation = GO_RED;
            } else {
                go_situation = GO_YELLOW;
            }
            break;
        case ALL_RED_BEFORE_NS_GREEN:
            go_situation = GO_RED;
            break;
        default:
            fprintf(stderr, "Invalid intersection state\n");
            exit(1);
    }

    return (GoSituationWithCountdown){go_situation, intersection->countdown};
}