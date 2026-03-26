#include "rl/core/rl_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ===========================================================================
// Observation normaliser
// ===========================================================================

ObsNormalizer obs_normalizer_make(int dim) {
    ObsNormalizer on;
    on.dim      = dim;
    on.mean     = (float*)calloc(dim, sizeof(float));
    on.var      = (float*)malloc(sizeof(float) * dim);
    for (int d = 0; d < dim; d++) on.var[d] = 1.0f;
    on.count    = 0.0f;
    on._buf_mean = (float*)malloc(sizeof(float) * dim);
    on._buf_var  = (float*)malloc(sizeof(float) * dim);
    on._shifts   = (float*)malloc(sizeof(float) * dim);
    on._scales   = (float*)malloc(sizeof(float) * dim);
    return on;
}

void obs_normalizer_free(ObsNormalizer* on) {
    free(on->mean);     on->mean     = NULL;
    free(on->var);      on->var      = NULL;
    free(on->_buf_mean); on->_buf_mean = NULL;
    free(on->_buf_var);  on->_buf_var  = NULL;
    free(on->_shifts);   on->_shifts   = NULL;
    free(on->_scales);   on->_scales   = NULL;
}

void obs_normalizer_update(ObsNormalizer* on, const float* obs, int batch) {
    int dim = on->dim;
    mat_col_mean_and_var(obs, batch, dim, on->_buf_mean, on->_buf_var);

    float tot = on->count + batch;
    for (int d = 0; d < dim; d++) {
        float delta    = on->_buf_mean[d] - on->mean[d];
        float new_mean = on->mean[d] + delta * batch / tot;
        float m_a = on->var[d] * on->count;
        float m_b = on->_buf_var[d] * batch;
        float m_2 = m_a + m_b + delta * delta * (on->count * batch / tot);
        on->mean[d] = new_mean;
        on->var[d]  = m_2 / tot;
    }
    on->count = tot;
}

void obs_normalizer_apply(const ObsNormalizer* on, float* obs, int batch) {
    int dim = on->dim;
    for (int d = 0; d < dim; d++) {
        on->_shifts[d] = -on->mean[d];
        on->_scales[d] = 1.0f / sqrtf(on->var[d] + 1e-8f);
    }
    mat_col_shift_and_scale(obs, batch, dim, on->_shifts, on->_scales);
    vec_clamp(obs, batch * dim, -10.0f, 10.0f);
}

// ===========================================================================
// Fisher–Yates shuffle
// ===========================================================================

void shuffle(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}
