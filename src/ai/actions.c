#include "ai.h"
#include "map.h"
#include "math.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CARS 1024
// All actions will output true or false to determine whether the procedure (action) completed (True) or is in progress (False)

// ────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────

/** Detects whether the car has fully cleared the upcoming / current
 *  intersection (i.e. the turn manoeuvre is finished). */
static bool _turn_finished(const SituationalAwareness *s) {
    /* We consider the turn finished once the car is neither ON an
     * intersection nor ABOUT TO ENTER one. */
    return !s->is_on_intersection && !s->is_an_intersection_upcoming;
}

/** Convenience: set desired cruise speed with the regular acceleration
 *  controller. */
static void _cruise_to_speed(Car *self, MetersPerSecond target) {
    MetersPerSecondSquared a = car_compute_acceleration_cruise(self, target);
    car_set_acceleration(self, a);
}

/** Determine indicator enum from textual direction spec. */
static CarIndictor _dir_to_indicator(const char *dir) {
    if (!dir) return INDICATOR_NONE;
    if (strcmp(dir, "left") == 0 || strcmp(dir, "l") == 0)  return INDICATOR_LEFT;
    if (strcmp(dir, "right") == 0 || strcmp(dir, "r") == 0) return INDICATOR_RIGHT;
    return INDICATOR_NONE;
}

// ────────────────────────────────────────────────────────────────
// Intersection turning
// ────────────────────────────────────────────────────────────────

bool left_turn(Car *self, Simulation *sim) {
    const SituationalAwareness *s = sim_get_situational_awareness(sim, self->id);

    // If we have already finished the turn, clear the indicator and return.
    if (_turn_finished(s)) {
        car_set_indicator_turn(self, INDICATOR_NONE);
        return true;
    }

    // Ensure we are indicating left.
    if (car_get_indicator_turn(self) != INDICATOR_LEFT)
        car_set_indicator_turn(self, INDICATOR_LEFT);

    // Approach / negotiate the turn at the driver’s preferred turn speed.
    _cruise_to_speed(self, self->preferences.turn_speed);
    return false; // still turning
}

bool right_turn(Car *self, Simulation *sim) {
    const SituationalAwareness *s = sim_get_situational_awareness(sim, self->id);

    if (_turn_finished(s)) {
        car_set_indicator_turn(self, INDICATOR_NONE);
        return true;
    }

    if (car_get_indicator_turn(self) != INDICATOR_RIGHT)
        car_set_indicator_turn(self, INDICATOR_RIGHT);

    _cruise_to_speed(self, self->preferences.turn_speed);
    return false;
}

// ────────────────────────────────────────────────────────────────
// Lane merging
// ────────────────────────────────────────────────────────────────

static double _merge_started_at[MAX_CARS] = {0};

bool merge(Car *self, Simulation *sim, const char *direction, double duration) {
    const SituationalAwareness *s = sim_get_situational_awareness(sim, self->id);

    CarIndictor ind = _dir_to_indicator(direction);
    if (ind == INDICATOR_NONE) return true;      // nothing requested → done

    /* ── check duration timer -----------─────────────────────────────── */
    int cid = car_get_id(self);
    if (cid >= MAX_CARS) cid = MAX_CARS - 1;  // clamp

    double now = sim_get_time(sim);
    if (_merge_started_at[cid] <= 0.0)
        _merge_started_at[cid] = now;          // first call this merge

    if (now - _merge_started_at[cid] > duration) {
        LOG_WARN("Car %d: aborting merge – timeout reached.", cid);
        car_set_indicator_lane(self, INDICATOR_NONE);
        _merge_started_at[cid] = 0;
        return true;                           // finished (unsuccessfully)
    }
    /* ──────────────────────────────────────────────────────────────── */

    const Lane *target_lane = s->lane_target_for_indicator[ind];
    if (!target_lane) {                        // no lane on that side
        car_set_indicator_lane(self, INDICATOR_NONE);
        _merge_started_at[cid] = 0;
        return true;
    }

    /* Already in the desired lane? */
    if (car_get_lane_id(self) == target_lane->id) {  // TODO: this is wrong. We need to correctly track the target lane specified in prev call and check if we are in it now.
        car_set_indicator_lane(self, INDICATOR_NONE);
        _merge_started_at[cid] = 0;            // reset timer
        return true;                           // merge complete
    }

    /* Safety check – wait if dangerous. */
    if (car_is_lane_change_dangerous(self, sim, s, ind)) {
        car_set_indicator_lane(self, ind);     // keep signalling
        car_set_request_indicated_lane(self, false); // don't request lane change yet
        return false;                          // still waiting
    }

    /* Safe: initiate lane change & hold cruise. */
    car_set_indicator_lane_and_request(self, ind);
    _cruise_to_speed(self,
        lane_get_speed_limit(target_lane) + self->preferences.average_speed_offset);
    return true;                              // For now, assume the sim will merge right away. TODO: correctly track the target lane previously specified and check if we are in it now.
}

// ────────────────────────────────────────────────────────────────
// Overtaking / passing
// ────────────────────────────────────────────────────────────────

static double _pass_started_at[MAX_CARS] = {0};

bool pass_vehicle(Car *self,
                  Simulation *sim,
                  Car *target,
                  bool should_merge,
                  Meters merge_buffer,
                  MetersPerSecond speed_buffer,
                  double duration)
{
    /* ─────────────────────  bookkeeping & time-out  ─────────────────── */
    double        merge_duration = duration * 0.3;      /* 30 % slice for each merge */
    if (!self || !target) return true;                   /* nothing to do           */

    int    cid = car_get_id(self);
    if (cid >= MAX_CARS) cid = MAX_CARS - 1;

    double now = sim_get_time(sim);
    if (_pass_started_at[cid] <= 0.0) _pass_started_at[cid] = now;

    if (now - _pass_started_at[cid] > duration) {
        LOG_WARN("Car %d: aborting pass – timeout reached.", cid);
        _pass_started_at[cid] = 0;
        return true;                                     /* failed / timed-out     */
    }
    /* ─────────────────────────────────────────────────────────────────── */

    const SituationalAwareness *s = sim_get_situational_awareness(sim, cid);

    /* STEP 0 – are we already well past the target?                      */
    if (car_get_lane_progress_meters(self) >
        car_get_lane_progress_meters(target) + car_get_length(self) + merge_buffer)
    {
        /* STEP 0a – merge back if requested and not yet in the target’s lane. */
        if (should_merge && car_get_lane_id(self) != car_get_lane_id(target))
        {
            /* Work laterally toward the original lane one step at a time.
             * We try whichever side actually moves us closer to the target,
             * even if that means switching sides compared with the initial pass. */
            const Lane *left_lane  = s->lane_left;
            const Lane *right_lane = s->lane_right;

            const char *dir_needed = NULL;

            int my_lane     = car_get_lane_id(self);
            int target_lane = car_get_lane_id(target);

            /* Prefer a directly adjacent move if possible … */
            if (left_lane  && left_lane->id  == target_lane) dir_needed = "left";
            if (right_lane && right_lane->id == target_lane) dir_needed = "right";

            /* … otherwise step laterally toward the target lane id. */
            if (!dir_needed) {
                if (target_lane < my_lane && left_lane)       dir_needed = "left";
                else if (target_lane > my_lane && right_lane) dir_needed = "right";
            }

            if (dir_needed && merge(self, sim, dir_needed, merge_duration))
            {
                /* Merge finished – maybe we need another hop, so stay “in progress”
                 * until we’re truly in the same lane.                                 */
                return car_get_lane_id(self) == target_lane
                         ? (_pass_started_at[cid] = 0, true)    /* finished */
                         : false;                                /* hop again */
            }

            /* Couldn’t merge this tick – keep trying. */
            return false;
        }

        _pass_started_at[cid] = 0;
        return true; /* pass completed, no merge requested */
    }

    /* STEP 1 – still behind target, move to an overtaking lane. */
    if (car_get_lane_id(self) == car_get_lane_id(target))
    {
        /* First try left, else right; swap automatically on later calls. */
        if (merge(self, sim, "left", merge_duration) || merge(self, sim, "right", merge_duration))
            ; /* merge() already handles its own progress/timeout */
        return false;
    }

    /* STEP 2 – in an overtaking lane: accelerate past target. */
    MetersPerSecond desired = car_get_speed(target) + speed_buffer;
    if (desired < from_mph(0.0)) {
        LOG_ERROR("Car %d: pass_vehicle called with negative desired speed (%f).", cid, desired);
        desired = from_mph(0.0); // avoid negative speeds
    }
    _cruise_to_speed(self, desired);
    return false;
}

// ────────────────────────────────────────────────────────────────
// Basic cruising helpers
// ────────────────────────────────────────────────────────────────
static double _cruise_continue_started_at[MAX_CARS] = {0};

bool cruise(Car *self, float target_speed) {
    _cruise_to_speed(self, target_speed);
    return fabs(car_get_speed(self) - target_speed) < from_mph(1.0); // done once within 1 mph
}

bool cruise_continue(Car *self, Simulation *sim, double duration) {
    /* ── check duration timer -----------─────────────────────────────── */
    int cid = car_get_id(self);
    if (cid >= MAX_CARS) cid = MAX_CARS - 1;  // clamp

    double now = sim_get_time(sim);
    if (_cruise_continue_started_at[cid] <= 0.0)
        _cruise_continue_started_at[cid] = now;          // first call this merge

    if (now - _cruise_continue_started_at[cid] > duration) {
        LOG_INFO("Car %d: aborting cruise_continue – timeout reached.", cid);
        car_set_indicator_lane(self, INDICATOR_NONE);
        _cruise_continue_started_at[cid] = 0;
        return true;                           // finished (unsuccessfully)
    }
    /* ──────────────────────────────────────────────────────────────── */
    // Set the acceleration to 0 to maintain current speed.
    MetersPerSecondSquared a = mpss(0.0);
    car_set_acceleration(self, a);
    // Keep whatever speed/acceleration we already have.
    _cruise_continue_started_at[cid] = 0;
    return true; // noop – always "finished"
}