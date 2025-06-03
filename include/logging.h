#pragma once

#include <stdbool.h>

typedef enum {
    LOG_ERROR = 0,
    LOG_WARN  = 1,
    LOG_INFO  = 2,
    LOG_DEBUG = 3,
    LOG_TRACE = 4
} LogLevel;

bool should_log(LogLevel level, const char *file);
void log_message(LogLevel level, const char *file, int line, const char *fmt, ...);

#define LOG(level, fmt, ...) do { \
    if (should_log(level, __FILE__)) { \
        log_message(level, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
    } \
} while (0)

#define LOG_ERROR(fmt, ...) LOG(LOG_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  LOG(LOG_WARN, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  LOG(LOG_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG(LOG_DEBUG, fmt, ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) LOG(LOG_TRACE, fmt, ##__VA_ARGS__)
