#include <stdbool.h>
#include <string.h>
#include "rl/core/nn.h"

// ===========================================================================
// Layer IO cache helper
// ===========================================================================

static void layer_cache_input(Layer* self, const float* input,
                              int batch_size, int dim) {
    if (!self->cache_io) return;
    if (batch_size > self->cached_batch_cap) {
        free(self->cached_input);
        self->cached_input = (float*)malloc(sizeof(float) * batch_size * dim);
        self->cached_batch_cap = batch_size;
    }
    memcpy(self->cached_input, input, sizeof(float) * batch_size * dim);
}

// ===========================================================================
// Linear layer implementation
// ===========================================================================

static void linear_forward(Layer* self, const float* input,
                           float* output, int batch_size) {
    LinearLayer* ll = (LinearLayer*)self;
    layer_cache_input(self, input, batch_size, ll->in_features);
    mat_mul_bias(output, input, ll->W->data, ll->b->data,
                 batch_size, ll->in_features, ll->out_features);
}

static void linear_backward(Layer* self, const float* d_output,
                             float* d_input, int batch_size) {
    LinearLayer* ll = (LinearLayer*)self;
    mat_col_sum_add(ll->grad_b->data, d_output, batch_size, ll->out_features);
    mat_mul_transA_add(ll->grad_W->data, self->cached_input, d_output,
                       batch_size, ll->in_features, ll->out_features);
    if (d_input)
        mat_mul_transB(d_input, d_output, ll->W->data,
                       batch_size, ll->out_features, ll->in_features);
}

static int  linear_in_dim(Layer* s)    { return ((LinearLayer*)s)->in_features; }
static int  linear_out_dim(Layer* s)   { return ((LinearLayer*)s)->out_features; }
static int  linear_num_params(Layer* s){ (void)s; return 2; }

static void linear_fill_params(Layer* self, Tensor** params, Tensor** grads) {
    LinearLayer* ll = (LinearLayer*)self;
    params[0] = ll->W;      grads[0] = ll->grad_W;
    params[1] = ll->b;      grads[1] = ll->grad_b;
}

static void linear_zero_grad(Layer* self) {
    LinearLayer* ll = (LinearLayer*)self;
    tensor_zero(ll->grad_W);
    tensor_zero(ll->grad_b);
}

static void linear_free_fn(Layer* self) {
    LinearLayer* ll = (LinearLayer*)self;
    tensor_free(ll->W);
    tensor_free(ll->b);
    tensor_free(ll->grad_W);
    tensor_free(ll->grad_b);
    free(self->cached_input);
    free(ll);
}

static const LayerVTable linear_vtable = {
    .forward     = linear_forward,
    .backward    = linear_backward,
    .in_dim      = linear_in_dim,
    .out_dim     = linear_out_dim,
    .num_params  = linear_num_params,
    .fill_params = linear_fill_params,
    .zero_grad   = linear_zero_grad,
    .free_fn     = linear_free_fn,
};

Layer* linear_layer_malloc(int in_features, int out_features) {
    LinearLayer* ll = (LinearLayer*)calloc(1, sizeof(LinearLayer));
    ll->base.type   = LAYER_LINEAR;
    ll->base.vtable = &linear_vtable;
    ll->in_features  = in_features;
    ll->out_features = out_features;

    // W: Xavier uniform  (tensor_malloc zero-inits, then we overwrite)
    ll->W = tensor2d_malloc(in_features, out_features);
    double limit = sqrt(6.0 / (in_features + out_features));
    for (int i = 0; i < in_features * out_features; i++)
        ll->W->data[i] = (float)rand_uniform(-limit, limit);

    // b: zeros  (already zero from tensor_malloc)
    ll->b = tensor1d_malloc(out_features);

    // Gradients: zeros
    ll->grad_W = tensor2d_malloc(in_features, out_features);
    ll->grad_b = tensor1d_malloc(out_features);

    return (Layer*)ll;
}

// ===========================================================================
// ReLU layer implementation
// ===========================================================================

static void relu_forward_fn(Layer* self, const float* input,
                            float* output, int batch_size) {
    ReLULayer* rl = (ReLULayer*)self;
    int size = batch_size * rl->dim;
    layer_cache_input(self, input, batch_size, rl->dim);
    memcpy(output, input, sizeof(float) * size);
    vec_relu(output, size);
}

static void relu_backward_fn(Layer* self, const float* d_output,
                              float* d_input, int batch_size) {
    ReLULayer* rl = (ReLULayer*)self;
    int size = batch_size * rl->dim;
    if (d_input) {
        memcpy(d_input, d_output, sizeof(float) * size);
        vec_relu_backward(d_input, self->cached_input, size);
    }
}

static int  relu_in_dim(Layer* s)    { return ((ReLULayer*)s)->dim; }
static int  relu_out_dim(Layer* s)   { return ((ReLULayer*)s)->dim; }
static int  relu_num_params(Layer* s){ (void)s; return 0; }
static void relu_fill_params(Layer* s, Tensor** p, Tensor** g)
    { (void)s; (void)p; (void)g; }
static void relu_zero_grad(Layer* s) { (void)s; }

static void relu_free_fn(Layer* self) {
    free(self->cached_input);
    free(self);
}

static const LayerVTable relu_vtable = {
    .forward     = relu_forward_fn,
    .backward    = relu_backward_fn,
    .in_dim      = relu_in_dim,
    .out_dim     = relu_out_dim,
    .num_params  = relu_num_params,
    .fill_params = relu_fill_params,
    .zero_grad   = relu_zero_grad,
    .free_fn     = relu_free_fn,
};

Layer* relu_layer_malloc(int dim) {
    ReLULayer* rl = (ReLULayer*)calloc(1, sizeof(ReLULayer));
    rl->base.type   = LAYER_RELU;
    rl->base.vtable = &relu_vtable;
    rl->dim = dim;
    return (Layer*)rl;
}

// ===========================================================================
// Sequential neural network
// ===========================================================================

static void seq_nn_build_views(SequentialNN* nn) {
    int n = 0;
    for (int i = 0; i < nn->num_layers; i++)
        n += nn->layers[i]->vtable->num_params(nn->layers[i]);
    nn->num_param_tensors = n;
    if (n == 0) { nn->param_views = NULL; nn->grad_views = NULL; return; }
    nn->param_views = (Tensor**)malloc(sizeof(Tensor*) * n);
    nn->grad_views  = (Tensor**)malloc(sizeof(Tensor*) * n);
    int idx = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        Layer* l = nn->layers[i];
        int np = l->vtable->num_params(l);
        if (np > 0) {
            l->vtable->fill_params(l, &nn->param_views[idx],
                                   &nn->grad_views[idx]);
            idx += np;
        }
    }
}

SequentialNN* seq_nn_malloc(int num_layers, Layer** layers, bool backprop) {
    SequentialNN* nn = (SequentialNN*)calloc(1, sizeof(SequentialNN));
    nn->num_layers = num_layers;
    nn->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
    memcpy(nn->layers, layers, sizeof(Layer*) * num_layers);
    nn->backprop = backprop;
    for (int i = 0; i < num_layers; i++)
        nn->layers[i]->cache_io = backprop;
    seq_nn_build_views(nn);
    return nn;
}

void seq_nn_forward(SequentialNN* nn, const float* input,
                    float* output, int batch_size) {
    if (nn->num_layers == 0) return;

    int max_dim = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        int d = layer_out_dim(nn->layers[i]);
        if (d > max_dim) max_dim = d;
    }

    float* buf[2] = {
        (float*)malloc(sizeof(float) * batch_size * max_dim),
        (float*)malloc(sizeof(float) * batch_size * max_dim),
    };

    const float* cur_in = input;
    for (int i = 0; i < nn->num_layers; i++) {
        float* cur_out = (i == nn->num_layers - 1)
                         ? output : buf[i % 2];
        layer_forward(nn->layers[i], cur_in, cur_out, batch_size);
        cur_in = cur_out;
    }

    free(buf[0]);
    free(buf[1]);
}

void seq_nn_backward(SequentialNN* nn, const float* d_output,
                     float* d_input_out, int batch_size) {
    if (nn->num_layers == 0) return;

    int max_dim = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        int d = layer_in_dim(nn->layers[i]);
        if (d > max_dim) max_dim = d;
        d = layer_out_dim(nn->layers[i]);
        if (d > max_dim) max_dim = d;
    }

    float* buf[2] = {
        (float*)malloc(sizeof(float) * batch_size * max_dim),
        (float*)malloc(sizeof(float) * batch_size * max_dim),
    };

    int last_out = layer_out_dim(nn->layers[nn->num_layers - 1]);
    memcpy(buf[0], d_output, sizeof(float) * batch_size * last_out);

    float* cur_d_out = buf[0];
    for (int i = nn->num_layers - 1; i >= 0; i--) {
        float* cur_d_in;
        if (i > 0)
            cur_d_in = (cur_d_out == buf[0]) ? buf[1] : buf[0];
        else
            cur_d_in = d_input_out;
        layer_backward(nn->layers[i], cur_d_out, cur_d_in, batch_size);
        cur_d_out = cur_d_in;
    }

    free(buf[0]);
    free(buf[1]);
}

void seq_nn_zero_grad(SequentialNN* nn) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_zero_grad(nn->layers[i]);
}

void seq_nn_copy_params(SequentialNN* dst, SequentialNN* src) {
    for (int i = 0; i < dst->num_param_tensors; i++)
        tensor_copy_data(dst->param_views[i], src->param_views[i]);
}

void seq_nn_set_backprop(SequentialNN* nn, bool backprop) {
    nn->backprop = backprop;
    for (int i = 0; i < nn->num_layers; i++)
        nn->layers[i]->cache_io = backprop;
}

SequentialNN* seq_nn_clone(SequentialNN* src, bool backprop) {
    int nl = src->num_layers;
    Layer** layers = (Layer**)malloc(sizeof(Layer*) * nl);
    for (int i = 0; i < nl; i++) {
        Layer* s = src->layers[i];
        if (s->type == LAYER_LINEAR) {
            LinearLayer* sl = (LinearLayer*)s;
            layers[i] = linear_layer_malloc(sl->in_features, sl->out_features);
        } else {
            layers[i] = relu_layer_malloc(layer_in_dim(s));
        }
    }
    SequentialNN* nn = seq_nn_malloc(nl, layers, backprop);
    free(layers);
    seq_nn_copy_params(nn, src);
    return nn;
}

void seq_nn_free(SequentialNN* nn) {
    if (!nn) return;
    for (int i = 0; i < nn->num_layers; i++)
        layer_free(nn->layers[i]);
    free(nn->layers);
    free(nn->param_views);   // shallow pointer array only
    free(nn->grad_views);
    free(nn);
}

SequentialNN* seq_nn_make_mlp(int num_sizes, const int* sizes, bool backprop) {
    int num_linear = num_sizes - 1;
    int num_relu   = (num_linear > 1) ? num_linear - 1 : 0;
    int total      = num_linear + num_relu;

    Layer** layers = (Layer**)malloc(sizeof(Layer*) * total);
    int idx = 0;
    for (int i = 0; i < num_linear; i++) {
        layers[idx++] = linear_layer_malloc(sizes[i], sizes[i + 1]);
        if (i < num_linear - 1)
            layers[idx++] = relu_layer_malloc(sizes[i + 1]);
    }

    SequentialNN* nn = seq_nn_malloc(total, layers, backprop);
    free(layers);
    return nn;
}

// ===========================================================================
// Adam optimizer
// ===========================================================================

AdamState* adam_malloc(Tensor** params, int num_params,
                      float lr, float beta1, float beta2, float eps) {
    AdamState* a = (AdamState*)malloc(sizeof(AdamState));
    a->num_params = num_params;
    a->lr = lr;
    a->beta1 = beta1;
    a->beta2 = beta2;
    a->eps = eps;
    a->m = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    a->v = (Tensor**)malloc(sizeof(Tensor*) * num_params);
    for (int i = 0; i < num_params; i++) {
        a->m[i] = tensor_malloc(params[i]->rank,
                                params[i]->n1, params[i]->n2,
                                params[i]->n3, params[i]->n4);
        a->v[i] = tensor_malloc(params[i]->rank,
                                params[i]->n1, params[i]->n2,
                                params[i]->n3, params[i]->n4);
    }
    a->t = 0;
    return a;
}

void adam_free(AdamState* a) {
    if (a) {
        for (int i = 0; i < a->num_params; i++) {
            tensor_free(a->m[i]);
            tensor_free(a->v[i]);
        }
        free(a->m);
        free(a->v);
        free(a);
    }
}

void adam_step(AdamState* a, Tensor** params, Tensor** grads) {
    float lr = a->lr, beta1 = a->beta1, beta2 = a->beta2, eps = a->eps;
    a->t++;
    float bc1 = 1 - powf(beta1, a->t);
    float bc2 = 1 - powf(beta2, a->t);

    for (int i = 0; i < a->num_params; i++) {
        int sz = tensor_size(params[i]);
        float* p = params[i]->data;
        float* g = grads[i]->data;
        float* m = a->m[i]->data;
        float* v = a->v[i]->data;

        for (int j = 0; j < sz; j++) {
            m[j] = beta1 * m[j] + (1 - beta1) * g[j];
            v[j] = beta2 * v[j] + (1 - beta2) * g[j] * g[j];
            float m_hat = m[j] / bc1;
            float v_hat = v[j] / bc2;
            p[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

// ===========================================================================
// Gradient clipping
// ===========================================================================

void grad_clip_norm(Tensor** grads, int n, float max_norm) {
    float sq = 0.0f;
    for (int i = 0; i < n; i++) {
        int sz = tensor_size(grads[i]);
        float* d = grads[i]->data;
        for (int j = 0; j < sz; j++) sq += d[j] * d[j];
    }
    float norm = sqrtf(sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; i++)
            tensor_scale(grads[i], scale);
    }
}
