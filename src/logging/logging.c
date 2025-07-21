#include "logging.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#define FILENO _fileno
#define ISATTY _isatty
#else
#include <unistd.h>
#define FILENO fileno
#define ISATTY isatty
#endif

static LogLevel current_log_level = LOG_DEBUG;
static const char *trace_files = NULL;
static bool is_initialized = false;
static bool use_color = false;

static const char *get_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) slash = strrchr(path, '\\');
    return slash ? slash + 1 : path;
}

static void initialize_logging(void) {
    const char *level_str = getenv("SIM_LOG_LEVEL");
    if (level_str) {
        if (strcmp(level_str, "ERROR") == 0) current_log_level = LOG_ERROR;
        else if (strcmp(level_str, "WARN") == 0) current_log_level = LOG_WARN;
        else if (strcmp(level_str, "INFO") == 0) current_log_level = LOG_INFO;
        else if (strcmp(level_str, "DEBUG") == 0) current_log_level = LOG_DEBUG;
        else if (strcmp(level_str, "TRACE") == 0) current_log_level = LOG_TRACE;
    }
    trace_files = getenv("SIM_TRACE");

    const char *color_str = getenv("SIM_LOG_COLOR");
    if (color_str) {
        if (strcmp(color_str, "always") == 0) {
            use_color = true;
        } else if (strcmp(color_str, "never") == 0) {
            use_color = false;
        } else if (strcmp(color_str, "auto") == 0) {
            use_color = ISATTY(FILENO(stderr));
        }
    } else {
        use_color = ISATTY(FILENO(stderr)); // Default to auto
    }

    is_initialized = true;
}

static bool is_file_traced(const char *file) {
    if (!trace_files) {
        return false;
    }
    const char *basename = get_basename(file);
    size_t len = strlen(basename);
    const char *p = trace_files;
    while (*p) {
        const char *semicolon = strchr(p, ';');
        size_t segment_len = semicolon ? (size_t)(semicolon - p) : strlen(p);
        if (segment_len == len && strncmp(p, basename, len) == 0) {
            return true;
        }
        p = semicolon ? semicolon + 1 : p + segment_len;
    }
    return false;
}

bool should_log(LogLevel level, const char *file) {
    if (!is_initialized) {
        initialize_logging();
    }
    if (level == LOG_TRACE) {
        return current_log_level >= LOG_TRACE || is_file_traced(file);
    }
    return level <= current_log_level;
}

void log_message(LogLevel level, const char *file, int line, const char *fmt, ...) {
    const char *level_str;
    const char *color_code = "";
    const char *reset_code = use_color ? "\033[0m" : "";

    if (use_color) {
        switch (level) {
            case LOG_ERROR: color_code = "\033[31m"; break;  // Red (non-bold)
            case LOG_WARN:  color_code = "\033[33m"; break;  // Yellow (non-bold)
            case LOG_INFO:  color_code = "\033[32m"; break;  // Green
            case LOG_DEBUG: color_code = "\033[35m"; break;  // Purple (non-bold)
            case LOG_TRACE: color_code = "\033[90m"; break;  // Gray (bright black)
            default:        color_code = "\033[0m"; break;   // Default
        }
    }

    switch (level) {
        case LOG_ERROR: level_str = "ERROR"; break;
        case LOG_WARN:  level_str = "WARN";  break;
        case LOG_INFO:  level_str = "INFO";  break;
        case LOG_DEBUG: level_str = "DEBUG"; break;
        case LOG_TRACE: level_str = "TRACE"; break;
        default:        level_str = "UNKNOWN"; break;
    }

    fprintf(stderr, "%s[%s] %s:%d: ", color_code, level_str, file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "%s\n", reset_code);
    fflush(stderr);
}