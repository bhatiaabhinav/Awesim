#include <stdlib.h>
#include <string.h>
#include "bad.h"
#include "logging.h"

static bool traffic_violations_valid_car_id(CarId car_id) {
    return car_id >= 0 && car_id < MAX_CARS_IN_SIMULATION;
}

TrafficViolationsLogsQueue* traffic_violations_logs_queue_malloc() {
    TrafficViolationsLogsQueue* queue = (TrafficViolationsLogsQueue*)malloc(sizeof(TrafficViolationsLogsQueue));
    if (!queue) {
        LOG_ERROR("Failed to allocate TrafficViolationsLogsQueue");
        return NULL;
    }
    traffic_violations_logs_queue_clear_all(queue);
    return queue;
}

void traffic_violations_logs_queue_free(TrafficViolationsLogsQueue* queue) {
    if (!queue) {
        LOG_ERROR("Attempted to free a NULL TrafficViolationsLogsQueue pointer");
        return;
    }
    free(queue);
}

TrafficViolation traffic_violation_get_for_car(TrafficViolationsLogsQueue* queue, CarId car_id, int violation_index_from_most_recent) {
    TrafficViolation empty = { .car_id = -1, .type = 0, .time = 0.0, .detail = 0.0 };

    if (!queue || !traffic_violations_valid_car_id(car_id)) return empty;

    int total = queue->num_violations[car_id];
    if (total == 0 || violation_index_from_most_recent < 0) return empty;

    // The number of violations still available in the circular buffer.
    int available = total < TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE ? total : TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE;
    if (violation_index_from_most_recent >= available) return empty;

    // next_index points one past the most recent write, so the most recent
    // entry is at (next_index - 1). Going back by violation_index_from_most_recent
    // more steps gives us the requested entry.
    int index = (queue->next_index[car_id] - 1 - violation_index_from_most_recent + TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE * 2) % TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE;
    return queue->queue[car_id][index];
}

void traffic_violation_log(TrafficViolationsLogsQueue* queue, CarId car_id, TrafficViolationType type, Seconds time, double detail) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return;
    if (!queue->enabled[car_id]) return;

    int idx = queue->next_index[car_id];
    queue->queue[car_id][idx] = (TrafficViolation){
        .car_id = car_id,
        .type = type,
        .time = time,
        .detail = detail,
    };
    queue->next_index[car_id] = (idx + 1) % TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE;
    queue->num_violations[car_id]++;
    if (queue->queue_num_items[car_id] < TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE) {
        queue->queue_num_items[car_id]++;
    }
}

void traffic_violations_logs_queue_clear_for_car(TrafficViolationsLogsQueue* queue, CarId car_id) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return;
    queue->next_index[car_id] = 0;
    queue->num_violations[car_id] = 0;
    queue->queue_num_items[car_id] = 0;
    memset(queue->queue[car_id], 0, sizeof(queue->queue[car_id]));
}

void traffic_violations_logs_queue_clear_all(TrafficViolationsLogsQueue* queue) {
    if (!queue) return;
    memset(queue->queue, 0, sizeof(queue->queue));
    memset(queue->next_index, 0, sizeof(queue->next_index));
    memset(queue->num_violations, 0, sizeof(queue->num_violations));
    memset(queue->queue_num_items, 0, sizeof(queue->queue_num_items));
}
