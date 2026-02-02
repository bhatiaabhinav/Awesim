#include "utils.h"
#include "logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>  // for uint32_t

// --- LCG PRNG Implementation ---
#define LCG_A 1664525
#define LCG_C 1013904223
#define LCG_M ((uint32_t)0xFFFFFFFF)

static uint32_t lcg_state = 1;  // default nonzero seed

void seed_rng(uint32_t seed) {
    if (seed == 0) seed = 1;  // avoid 0 (common LCG pitfall)
    lcg_state = seed;
}

static long long counter = 0;    // Number of calls to lcg_rand for logging purposes
uint32_t lcg_rand() {
    lcg_state = (uint32_t)(LCG_A * lcg_state + LCG_C);  // Explicit cast to ensure 32-bit arithmetic
    LOG_TRACE("%lld LCG generated random number: %u", counter++, lcg_state);
    return lcg_state;
}

uint32_t lcg_get_state() {
    return lcg_state;
}

// --- Utility Functions ---

bool approxeq(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

double rand_0_to_1(void) {
    return (double)lcg_rand() / (double)UINT32_MAX;
}

double rand_uniform(double min, double max) {
    return min + (max - min) * rand_0_to_1();
}

int rand_int_range(int min_inclusive, int max_inclusive) {
    int range = max_inclusive - min_inclusive + 1;
    if (range <= 0) {
        fprintf(stderr, "rand_int_range: Error: max_inclusive must be greater than or equal to min_inclusive\n");
        exit(EXIT_FAILURE);
    }
    return min_inclusive + (int)(rand_0_to_1() * range);
}

double rand_gaussian(double mean, double std_dev) {

    // just return mean if std_dev is zero or negative
    if (std_dev <= 1e-9) {
        return mean;
    }

    static bool has_spare = false;
    static double spare;

    if (has_spare) {
        has_spare = false;
        return mean + std_dev * spare;
    }

    has_spare = true;
    double u, v, s;
    do {
        u = (rand_0_to_1() * 2.0) - 1.0;
        v = (rand_0_to_1() * 2.0) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std_dev * (u * s);
}

int rand_int(int max_exclusive) {
    if (max_exclusive <= 0) {
        fprintf(stderr, "rand_int: Error: max_exclusive must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return lcg_rand() % max_exclusive;
}

double fclamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}