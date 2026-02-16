#include <stdlib.h>
#include <string.h>
#include "bad.h"
#include "logging.h"

static bool traffic_violations_valid_car_id(CarId car_id) {
    return car_id >= 0 && car_id < MAX_CARS_IN_SIMULATION;
}

void traffic_violations_set_enabled(TrafficViolationsLogsQueue* queue, CarId car_id, bool enabled) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return;
    queue->enabled[car_id] = enabled;
}

void traffic_violations_logs_queue_free(TrafficViolationsLogsQueue* queue) {
    if (!queue) {
        LOG_ERROR("Attempted to free a NULL TrafficViolationsLogsQueue pointer");
        return;
    }
    free(queue);
}

TrafficViolation traffic_violation_get(TrafficViolationsLogsQueue* queue, CarId car_id, int violation_index_from_most_recent) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) {
        return (TrafficViolation){.car_id = ID_NULL}; // Return empty struct on error
    }
    
    int num_items = queue->queue_num_items[car_id];
    if (violation_index_from_most_recent < 0 || violation_index_from_most_recent >= num_items) {
        LOG_ERROR("Violation index %d out of bounds for car ID %d with %d logged violations", violation_index_from_most_recent, car_id, num_items);
        return (TrafficViolation){.car_id = ID_NULL}; // Index out of bounds
    }

    int next_idx = queue->next_index[car_id];
    // Calculate index in circular buffer: (next_index - 1 - k + SIZE) % SIZE
    int idx = (next_idx - 1 - violation_index_from_most_recent + TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE) % TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE;
    
    // Handle case where calculation might result in negative value before modulo (though +SIZE prevents this for small k)
    if (idx < 0) idx += TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE; 

    return queue->queue[car_id][idx];
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
    
    // increment number of violations of each type
    queue->num_violations_each_type[car_id][type]++;
    
    // Update number of items in the queue (capped at max size)
    if (queue->queue_num_items[car_id] < TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE) {
        queue->queue_num_items[car_id]++;
    }
}

void traffic_violations_logs_queue_clear(TrafficViolationsLogsQueue* queue, CarId car_id) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return;
    queue->next_index[car_id] = 0;
    queue->queue_num_items[car_id] = 0;
    memset(queue->queue[car_id], 0, sizeof(queue->queue[car_id]));
}

int traffic_violation_get_total_count(TrafficViolationsLogsQueue* queue, CarId car_id) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return 0;
    
    int total = 0;
    for (int type = 0; type < TRAFFIC_VIOLATION_NUM_TYPES; type++) {
        total += queue->num_violations_each_type[car_id][type];
    }
    return total;
}

int traffic_violation_type_get_total_count(TrafficViolationsLogsQueue* queue, CarId car_id, TrafficViolationType type) {
    if (!queue || !traffic_violations_valid_car_id(car_id)) return 0;
    if (type < 0 || type >= TRAFFIC_VIOLATION_NUM_TYPES) return 0;

    return queue->num_violations_each_type[car_id][type];
}
