#include "sim.h"
#include "logging.h"
#include "ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>


typedef double MetersCubedPerSecondsSquared;  // for ∫ |v|^3 dt

// -----------------------------------------------------------------------------
// Engine and Environment Configuration
// -----------------------------------------------------------------------------

// EngineEnv represents environmental and engine-related parameters used to
// adjust the simulation for different driving styles or conditions.
typedef struct {
    bool   conservative;          // True = conservative assumptions
    double driveline_eff;         // Wheel-to-engine mechanical path efficiency
    double accessory_power_W;     // Baseline accessory draw (ECU, alternator, etc.)
    double hvac_power_W;          // HVAC draw (A/C, cabin fan, etc.)
    bool   enable_fuel_cut;       // Enable deceleration fuel cut
    double fuel_cut_speed_gate;   // Speed threshold for fuel cut (m/s)
    double crawl_speed_gate;      // Enforce idle consumption below this speed (m/s)
    double fuel_density;          // Fuel density (kg/L)
    double bsfc_scale;            // Scaling factor for BSFC curve (>1 = worse)
    double added_grade;           // Continuous uphill grade as fraction (e.g., 0.005 = 0.5%)
    double headwind;              // Constant headwind speed (m/s)
} EngineEnv;

// -----------------------------------------------------------------------------
// Default environment initializer
// -----------------------------------------------------------------------------
// conservative = false: baseline, optimistic parameters
// conservative = true: realistic or slightly pessimistic assumptions
// -----------------------------------------------------------------------------
static inline EngineEnv default_env(bool conservative) {
    EngineEnv e;
    e.conservative        = conservative;
    e.driveline_eff       = conservative ? 0.88 : 0.90;
    e.accessory_power_W   = conservative ? 1200.0 : 800.0;   // Watts
    e.hvac_power_W        = conservative ? 500.0  : 0.0;     // Watts
    e.enable_fuel_cut     = true;
    e.fuel_cut_speed_gate = conservative ? 12.0 : 8.0;       // Speed threshold
    e.crawl_speed_gate    = 1.0;
    e.fuel_density        = 0.745;                           // kg/L gasoline
    e.bsfc_scale          = conservative ? 1.10 : 1.00;      // +10% worse BSFC
    e.added_grade         = conservative ? 0.005 : 0.0;      // 0.5% uphill
    e.headwind            = conservative ? 2.0   : 0.0;      // 2 m/s headwind
    return e;
}

// -----------------------------------------------------------------------------
// Brake Specific Fuel Consumption (BSFC) model
// -----------------------------------------------------------------------------
// Returns BSFC in g/kWh given brake power in kW. BSFC represents how many
// grams of fuel are consumed per kilowatt-hour of engine output.
//
// Typical gasoline engine BSFC:
// - High load: 280-330 g/kWh (efficient region)
// - Low load: 400-600 g/kWh (inefficient)
// -----------------------------------------------------------------------------
static inline double bsfc_g_per_kwh(double P_brake_kW, const EngineEnv* env) {
    double base;
    if (P_brake_kW < 2.0)        base = 650.0;
    else if (P_brake_kW < 5.0)   base = 550.0;
    else if (P_brake_kW < 10.0)  base = 430.0;
    else if (P_brake_kW < 20.0)  base = 340.0;
    else if (P_brake_kW < 40.0)  base = 285.0;
    else if (P_brake_kW < 70.0)  base = 305.0;
    else                          base = 335.0;

    return base * env->bsfc_scale;
}

// -----------------------------------------------------------------------------
// Core fuel consumption routine
// -----------------------------------------------------------------------------
// This function calculates the fuel consumed over a time interval given the
// vehicle state and environment. It models rolling resistance, aerodynamic
// drag, engine losses, and idle/deceleration fuel behavior.
// -----------------------------------------------------------------------------
Liters fuel_consumption_compute_env(MetersPerSecond initial_velocity,
                                    MetersPerSecondSquared accel,
                                    Seconds delta_time,
                                    bool include_idle,
                                    Kilograms mass,
                                    double rolling_resistance_coefficient,
                                    double drag_coefficient,
                                    MetersSquared frontal_area,
                                    KgPerMetersCubed air_density,
                                    MetersPerSecondSquared gravity,
                                    double engine_efficiency,               // ignored
                                    JoulesPerLiter fuel_energy_density,     // ignored
                                    LitersPerSecond idle_rate,
                                    const EngineEnv* env)
{
    // Validate inputs
    if (delta_time <= 0.0 || mass < 0.0 || frontal_area < 0.0 || air_density < 0.0 ||
        gravity < 0.0 || rolling_resistance_coefficient < 0.0 || drag_coefficient < 0.0 ||
        idle_rate < 0.0) {
        return 0.0;
    }

    // Prevent numerical overflow
    double max_velocity = fmax(fabs(initial_velocity), fabs(initial_velocity + accel * delta_time));
    if (max_velocity > 1e6) return 0.0;

    // --- Kinematics ---
    MetersPerSecond final_velocity = initial_velocity + accel * delta_time;
    Meters displacement = initial_velocity * delta_time + 0.5 * accel * delta_time * delta_time;

    // Change in kinetic energy
    Joules delta_kinetic_energy = 0.5 * mass * (final_velocity * final_velocity - initial_velocity * initial_velocity);

    // --- Compute integrals for rolling resistance and aerodynamic drag ---
    Meters integral_abs_velocity_dt = 0.0;               // ∫|v| dt
    MetersCubedPerSecondsSquared integral_abs_velocity_cubed_dt = 0.0; // ∫ v_air^2 * |v| dt

    const double eps = 1e-10;
    double velocity_scale = fmax(fabs(initial_velocity), fabs(final_velocity));
    double rel_eps = fmax(eps, velocity_scale * eps);

    // Determine velocity direction
    double sign_initial = (initial_velocity > rel_eps) ? 1.0 : ((initial_velocity < -rel_eps) ? -1.0 : 0.0);
    double sign_final   = (final_velocity > rel_eps)   ? 1.0 : ((final_velocity < -rel_eps)   ? -1.0 : 0.0);
    bool velocity_crosses_zero = (initial_velocity * final_velocity < 0.0) && (sign_initial != sign_final);

    // Constant velocity
    if (fabs(accel) < rel_eps) {
        double v = fabs(initial_velocity);
        integral_abs_velocity_dt = v * delta_time;

        // Account for headwind in aerodynamic drag
        double v_air = fabs(initial_velocity) + env->headwind;
        if (v_air < 0.0) v_air = 0.0;
        integral_abs_velocity_cubed_dt = v_air * v_air * v * delta_time;
    }
    // Simple segment with no zero crossing
    else if (!velocity_crosses_zero) {
        double sign_val = (sign_initial != 0.0) ? sign_initial : sign_final;
        integral_abs_velocity_dt = sign_val * displacement;

        // Approximate aerodynamic drag with midpoint air-relative speed
        double v_mid = 0.5 * (initial_velocity + final_velocity);
        double v_air_mid = fabs(v_mid) + env->headwind;
        if (v_air_mid < 0.0) v_air_mid = 0.0;
        integral_abs_velocity_cubed_dt = v_air_mid * v_air_mid * fabs(v_mid) * delta_time;
    }
    // Velocity crosses zero, split into two intervals
    else {
        Seconds t0 = -initial_velocity / accel;
        if (t0 < 0.0 || t0 > delta_time) return 0.0;

        // First segment
        Seconds dt1 = t0;
        Meters displacement1 = initial_velocity * dt1 + 0.5 * accel * dt1 * dt1;
        double v_mid1 = initial_velocity + 0.5 * accel * dt1;
        double v_air_mid1 = fabs(v_mid1) + env->headwind;
        if (v_air_mid1 < 0.0) v_air_mid1 = 0.0;

        // Second segment
        Seconds dt2 = delta_time - t0;
        Meters displacement2 = 0.5 * accel * dt2 * dt2;
        double v_mid2 = 0.5 * accel * dt2;
        double v_air_mid2 = fabs(v_mid2) + env->headwind;
        if (v_air_mid2 < 0.0) v_air_mid2 = 0.0;

        integral_abs_velocity_dt = sign_initial * displacement1 + sign_final * displacement2;
        integral_abs_velocity_cubed_dt = (v_air_mid1 * v_air_mid1 * fabs(v_mid1)) * dt1 +
                                         (v_air_mid2 * v_air_mid2 * fabs(v_mid2)) * dt2;
    }

    // --- Road load calculations ---
    // Rolling resistance and grade
    double F_rr = rolling_resistance_coefficient * mass * gravity;
    double F_grade = mass * gravity * fmax(env->added_grade, 0.0);
    Joules work_rr_grade = (F_rr + F_grade) * fmax(integral_abs_velocity_dt, 0.0);

    // Aerodynamic drag
    Joules work_aero = 0.5 * drag_coefficient * frontal_area * air_density *
                       fmax(integral_abs_velocity_cubed_dt, 0.0);

    // Total wheel mechanical energy
    Joules mechanical_energy = delta_kinetic_energy + work_rr_grade + work_aero;
    if (mechanical_energy < 0.0) mechanical_energy = 0.0;

    // Wheel power (average over the timestep)
    double P_wheel_W = mechanical_energy / delta_time;

    // Engine brake power = wheel power / driveline efficiency + accessories
    double P_driveline_W = P_wheel_W / env->driveline_eff;
    double P_brake_W = P_driveline_W + env->accessory_power_W + env->hvac_power_W;

    // --- Fuel consumption logic ---
    Liters fuel_consumed_L = 0.0;

    if (P_wheel_W <= 0.0) {
        // Vehicle coasting or braking
        if (env->enable_fuel_cut &&
            (fabs(initial_velocity) > env->fuel_cut_speed_gate || fabs(final_velocity) > env->fuel_cut_speed_gate)) {
            // Fuel cut enabled: assume backdrive covers accessory loads
        } else if (include_idle) {
            fuel_consumed_L += idle_rate * delta_time;
        }
    } else {
        // Engine under positive load: calculate fuel flow using BSFC
        double P_brake_kW = P_brake_W * 1e-3;                // Watts -> kW
        double bsfc = bsfc_g_per_kwh(P_brake_kW, env);       // g/kWh
        double fuel_mass_flow_kg_s = (bsfc * P_brake_kW) / (1000.0 * 3600.0);
        double fuel_vol_flow_L_s = fuel_mass_flow_kg_s / env->fuel_density;
        fuel_consumed_L += fuel_vol_flow_L_s * delta_time;
    }

    // Idle floor for very low speeds
    if (fabs(initial_velocity) < env->crawl_speed_gate && fabs(final_velocity) < env->crawl_speed_gate) {
        if (include_idle) {
            double idle_L = idle_rate * delta_time;
            if (fuel_consumed_L < idle_L) fuel_consumed_L = idle_L;
        }
    }

    return fuel_consumed_L;
}

// -----------------------------------------------------------------------------
// Wrapper for a "typical sedan" with fixed default vehicle parameters
// -----------------------------------------------------------------------------
Liters fuel_consumption_compute_typical_env(MetersPerSecond initial_velocity,
                                            MetersPerSecondSquared accel,
                                            Seconds delta_time,
                                            const EngineEnv* env)
{
    const bool include_idle = true;
    const Kilograms mass = 1500.0;        // kg
    const double c_rr = 0.015;            // rolling resistance coefficient
    const double c_d  = 0.30;             // drag coefficient
    const MetersSquared A = 2.2;          // frontal area (m^2)
    const KgPerMetersCubed rho_air = 1.225; // air density (kg/m^3)
    const MetersPerSecondSquared g = 9.81;  // gravity

    const double engine_efficiency = 0.25;          // legacy, unused
    const JoulesPerLiter fuel_energy_density = 32e6;// legacy, unused

    // Idle rate: 1.0 L/h in conservative mode, 0.8 L/h otherwise
    const LitersPerSecond idle_rate = env->conservative ? (1.0 / 3600.0) : 0.000222;

    return fuel_consumption_compute_env(initial_velocity, accel, delta_time, include_idle,
                                        mass, c_rr, c_d, A, rho_air, g,
                                        engine_efficiency, fuel_energy_density, idle_rate, env);
}

// -----------------------------------------------------------------------------
// Wrapper that defaults to conservative environment
// -----------------------------------------------------------------------------
Liters fuel_consumption_compute_typical(MetersPerSecond initial_velocity,
                                        MetersPerSecondSquared accel,
                                        Seconds delta_time)
{
    EngineEnv env = default_env(true); // conservative = true
    return fuel_consumption_compute_typical_env(initial_velocity, accel, delta_time, &env);
}
