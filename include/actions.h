#pragma once

/*
 * High-level driving actions for NPC and scripted AI cars.
 * Each routine returns `true` once the manoeuvre is finished
 * (successfully or aborted) and `false` while it is still in progress.
 *
 * The implementation lives in actions.c.
 */

#include "ai.h"     /* Car, Simulation, numeric units, enums, …              */
#include "map.h"    /* Lane, road geometry utilities                         */
#include "utils.h"  /* Misc helpers, logging macros, rand_0_to_1(), …        */

/* ──────────────────────────────────────────────────────────────── */
/* Intersection turning                                             */
/* ──────────────────────────────────────────────────────────────── */

/** Execute a left-turn through an intersection. */
bool left_turn(Car *self, Simulation *sim);

/** Execute a right-turn through an intersection. */
bool right_turn(Car *self, Simulation *sim);

/* ──────────────────────────────────────────────────────────────── */
/* Lane merging                                                     */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Initiate or continue a lane-merge manoeuvre.
 *
 * @param direction  "left", "l", "right", or "r" (case-insensitive).
 * @param duration   Hard time-out (seconds) before the merge aborts.
 */
bool merge(Car *self,
           Simulation *sim,
           const char *direction,
           double duration);

/* ──────────────────────────────────────────────────────────────── */
/* Overtaking / passing                                             */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Pass a slower vehicle and (optionally) merge back.
 *
 * @param target         Vehicle to overtake.
 * @param should_merge   Merge back into the original lane afterwards?
 * @param merge_buffer   Longitudinal gap to leave before merging back (m).
 * @param speed_buffer   Extra speed relative to @p target while passing (m/s).
 * @param duration       Hard time-out (seconds) for the entire manoeuvre.
 */
bool pass_vehicle(Car *self,
                  Simulation *sim,
                  Car *target,
                  bool should_merge,
                  Meters merge_buffer,
                  MetersPerSecond speed_buffer,
                  double duration);

/* ──────────────────────────────────────────────────────────────── */
/* Basic cruising helpers                                           */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Cruise toward a target speed.
 *
 * Returns `true` once the ego vehicle is within ±1 mph of the target.
 */
bool cruise(Car *self, float target_speed);

/**
 * Maintain current speed for a fixed duration (no-op helper).
 *
 * Useful for stitching actions in behaviour scripts.
 */
bool cruise_continue(Car *self, Simulation *sim, double duration);

