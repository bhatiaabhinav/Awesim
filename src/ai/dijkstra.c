#include "ai.h"

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ---------- Graph ---------- */

static int dg_valid_node(const DirectedGraph* g, int node) {
    return (g != NULL && node >= 0 && node < g->num_nodes);
}

DirectedGraph* dg_create(int max_nodes) {
    DirectedGraph* g;
    size_t n;

    if (max_nodes <= 0) return NULL;

    g = (DirectedGraph*)malloc(sizeof(DirectedGraph));
    if (!g) return NULL;

    g->num_nodes = 0;       /* start empty */
    g->max_nodes = max_nodes;

    n = (size_t)max_nodes;
    g->adj = (DG_Edge**)calloc(n, sizeof(DG_Edge*));
    if (!g->adj) {
        free(g);
        return NULL;
    }
    return g;
}

int dg_add_node(DirectedGraph* g) {
    int id;
    if (!g) return -1;
    if (g->num_nodes >= g->max_nodes) return -1;

    id = g->num_nodes;
    g->num_nodes += 1;
    /* g->adj[id] is already NULL because adj array was calloc'ed */
    return id;
}

static void dg_free_edges(DG_Edge* e) {
    while (e) {
        DG_Edge* next = e->next;
        free(e);
        e = next;
    }
}

void dg_free(DirectedGraph* g) {
    int i;
    if (!g) return;
    if (g->adj) {
        for (i = 0; i < g->num_nodes; i++) {
            dg_free_edges(g->adj[i]);
        }
        free(g->adj);
    }
    free(g);
}

int dg_add_edge_w(DirectedGraph* g, int from, int to, double weight) {
    DG_Edge* e;

    if (!dg_valid_node(g, from) || !dg_valid_node(g, to)) return 0;
    if (weight < 0.0) return 0; /* Dijkstra needs non-negative */

    e = (DG_Edge*)malloc(sizeof(DG_Edge));
    if (!e) return 0;

    e->to = to;
    e->weight = weight;
    e->next = g->adj[from];
    g->adj[from] = e;
    return 1;
}


int dg_add_edge(DirectedGraph* g, int from, int to) {
    return dg_add_edge_w(g, from, to, 1.0);
}


/* ---------- Min-Heap (Priority Queue) ---------- */

typedef struct {
    int node;
    double f_cost; // Priority (f = g + h)
    double g_cost; // Actual cost (g)
} DG_HeapItem;


typedef struct {
    DG_HeapItem* a;
    int size;
    int cap;
} DG_MinHeap;

static DG_MinHeap* dg_heap_create(int cap) {
    DG_MinHeap* h;
    if (cap <= 0) cap = 16;

    h = (DG_MinHeap*)malloc(sizeof(DG_MinHeap));
    if (!h) return NULL;

    h->a = (DG_HeapItem*)malloc((size_t)cap * sizeof(DG_HeapItem));
    if (!h->a) {
        free(h);
        return NULL;
    }
    h->size = 0;
    h->cap = cap;
    return h;
}

static void dg_heap_free(DG_MinHeap* h) {
    if (!h) return;
    free(h->a);
    free(h);
}

static void dg_heap_swap(DG_HeapItem* x, DG_HeapItem* y) {
    DG_HeapItem t = *x;
    *x = *y;
    *y = t;
}

static int dg_heap_push(DG_MinHeap* h, int node, double f_cost, double g_cost) {
    int i, p;

    if (!h) return 0;

    if (h->size == h->cap) {
        int newcap = h->cap * 2;
        DG_HeapItem* na = (DG_HeapItem*)realloc(h->a, (size_t)newcap * sizeof(DG_HeapItem));
        if (!na) return 0;
        h->a = na;
        h->cap = newcap;
    }

    i = h->size++;
    h->a[i].node = node;
    h->a[i].f_cost = f_cost;
    h->a[i].g_cost = g_cost;

    while (i > 0) {
        p = (i - 1) / 2;
        if (h->a[p].f_cost <= h->a[i].f_cost) break;
        dg_heap_swap(&h->a[p], &h->a[i]);
        i = p;
    }
    return 1;
}

static int dg_heap_empty(const DG_MinHeap* h) {
    return (h == NULL || h->size == 0);
}

static DG_HeapItem dg_heap_pop(DG_MinHeap* h) {
    DG_HeapItem top;
    int i;

    top = h->a[0];
    h->a[0] = h->a[--h->size];

    i = 0;
    for (;;) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int m = i;

        if (l < h->size && h->a[l].f_cost < h->a[m].f_cost) m = l;
        if (r < h->size && h->a[r].f_cost < h->a[m].f_cost) m = r;
        if (m == i) break;

        dg_heap_swap(&h->a[i], &h->a[m]);
        i = m;
    }
    return top;
}

/* ---------- AI / Dijkstra Path ---------- */

int dg_astar_path(const DirectedGraph* g,
                     int start,
                     int end,
                     DG_Heuristic h_func,
                     void* h_data,
                     int* node_array_out,
                     int node_array_len_out,
                     double* cost_out)
{
    double* dist; // Stores g_cost
    int* prev;
    unsigned char* visited;
    DG_MinHeap* pq;

    const double INF = (DBL_MAX / 4.0);
    const double EPS = 1e-12; /* for stale-entry checks */
    int i;

    if (cost_out) *cost_out = DBL_MAX;

    if (!g || !node_array_out || node_array_len_out <= 0) return -1;
    if (!dg_valid_node(g, start) || !dg_valid_node(g, end)) return -1;

    dist = (double*)malloc((size_t)g->num_nodes * sizeof(double));
    prev = (int*)malloc((size_t)g->num_nodes * sizeof(int));
    visited = (unsigned char*)calloc((size_t)g->num_nodes, 1);

    if (!dist || !prev || !visited) {
        free(dist);
        free(prev);
        free(visited);
        return -1;
    }

    for (i = 0; i < g->num_nodes; i++) {
        dist[i] = INF;
        prev[i] = -1;
    }
    dist[start] = 0.0;

    pq = dg_heap_create(32);
    if (!pq) {
        free(dist);
        free(prev);
        free(visited);
        return -1;
    }
    
    // Initial push with f = h(start, end)
    double start_h = (h_func) ? h_func(start, end, h_data) : 0.0;
    if (!dg_heap_push(pq, start, start_h, 0.0)) {
        dg_heap_free(pq);
        free(dist);
        free(prev);
        free(visited);
        return -1;
    }

    while (!dg_heap_empty(pq)) {
        DG_HeapItem it = dg_heap_pop(pq);
        int u = it.node;

        /* stale entry? Check against g_cost in dist array */
        {
            double du = dist[u];
            double tol = EPS * (1.0 + fabs(du));
            if (it.g_cost > du + tol) continue;
        }

        if (visited[u]) continue;
        visited[u] = 1;

        if (u == end) break;

        {
            DG_Edge* e;
            for (e = g->adj[u]; e != NULL; e = e->next) {
                int v = e->to;
                double w = e->weight;

                if (w < 0.0) continue;
                if (dist[u] >= INF) continue;

                {
                    double new_g = dist[u] + w;
                    if (new_g < dist[v]) {
                        dist[v] = new_g;
                        prev[v] = u;
                        
                        double h_val = (h_func) ? h_func(v, end, h_data) : 0.0;
                        double new_f = new_g + h_val;

                        if (!dg_heap_push(pq, v, new_f, new_g)) {
                            dg_heap_free(pq);
                            free(dist);
                            free(prev);
                            free(visited);
                            return -1;
                        }
                    }
                }
            }
        }
    }

    dg_heap_free(pq);

    if (dist[end] >= INF) {
        /* no path */
        free(dist);
        free(prev);
        free(visited);
        if (cost_out) *cost_out = DBL_MAX;
        return 0;
    }

    if (cost_out) *cost_out = dist[end];

    /* count path length by walking prev[] backwards */
    {
        int path_len = 0;
        int cur = end;

        while (cur != -1) {
            path_len++;
            if (cur == start) break;
            cur = prev[cur];
        }

        /* ensure it reaches start */
        if (cur != start) {
            free(dist);
            free(prev);
            free(visited);
            if (cost_out) *cost_out = DBL_MAX;
            return 0;
        }

        if (path_len > node_array_len_out) {
            free(dist);
            free(prev);
            free(visited);
            return -2;
        }

        /* write reversed into output */
        cur = end;
        i = path_len - 1;
        while (cur != -1) {
            node_array_out[i--] = cur;
            if (cur == start) break;
            cur = prev[cur];
        }

        free(dist);
        free(prev);
        free(visited);
        return path_len;
    }
}

int dg_dijkstra_path(const DirectedGraph* g,
                     int start,
                     int end,
                     int* node_array_out,
                     int node_array_len_out,
                     double* cost_out) {
    // Dijkstra is just A* with heuristic = 0
    return dg_astar_path(g, start, end, NULL, NULL, node_array_out, node_array_len_out, cost_out);
}


