#include "rl/core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// Ensure all parent directories for `path` exist.
static void ensure_parent_dirs(const char* path) {
    if (!path || !*path) return;

    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);

    size_t start = 0;
#ifdef _WIN32
    if (n >= 2 && buf[1] == ':') start = 2;
#endif

    for (size_t i = start; i < n; i++) {
        if (buf[i] != '/' && buf[i] != '\\') continue;
        char saved = buf[i];
        buf[i] = '\0';
        if (buf[0] != '\0') {
#ifdef _WIN32
            _mkdir(buf);
#else
            mkdir(buf, 0777);
#endif
        }
        buf[i] = saved;
    }
}

// Find column index by name, returns -1 if not found.
static int find_col(const CSVLogger* logger, const char* name) {
    for (int i = 0; i < logger->num_cols; i++) {
        if (strcmp(logger->col_names[i], name) == 0)
            return i;
    }
    return -1;
}

CSVLogger csv_logger(const char* filename, int flush_interval,
                     int num_cols, const char* col_names[]) {
    CSVLogger logger;
    memset(&logger, 0, sizeof(logger));
    logger.flush_interval   = flush_interval;
    logger.rows_since_flush = 0;

    // Store column names (capped at CSV_LOGGER_MAX_COLS)
    logger.num_cols = num_cols < CSV_LOGGER_MAX_COLS ? num_cols : CSV_LOGGER_MAX_COLS;
    for (int i = 0; i < logger.num_cols; i++) {
        size_t len = strlen(col_names[i]);
        logger.col_names[i] = (char*)malloc(len + 1);
        memcpy(logger.col_names[i], col_names[i], len + 1);
    }

    // Reset pending row
    for (int i = 0; i < logger.num_cols; i++) {
        logger.row_values[i] = 0.0;
        logger.row_set[i]    = false;
        logger.row_is_int[i] = false;
    }

    ensure_parent_dirs(filename);
    logger.fp = fopen(filename, "w");
    if (!logger.fp) {
        fprintf(stderr, "csv_logger: cannot open '%s' for writing\n", filename);
        return logger;
    }

    // Write header immediately
    for (int i = 0; i < logger.num_cols; i++) {
        if (i > 0) fputc(',', logger.fp);
        fputs(logger.col_names[i], logger.fp);
    }
    fputc('\n', logger.fp);
    fflush(logger.fp);

    return logger;
}

void csv_logger_log(CSVLogger* logger, int col, double value) {
    if (col < 0 || col >= logger->num_cols) return;
    logger->row_values[col] = value;
    logger->row_set[col]    = true;
    logger->row_is_int[col] = false;
}

void csv_logger_log_by_name(CSVLogger* logger, const char* name, double value) {
    csv_logger_log(logger, find_col(logger, name), value);
}

void csv_logger_log_int(CSVLogger* logger, int col, int value) {
    if (col < 0 || col >= logger->num_cols) return;
    logger->row_values[col] = (double)value;
    logger->row_set[col]    = true;
    logger->row_is_int[col] = true;
}

void csv_logger_log_int_by_name(CSVLogger* logger, const char* name, int value) {
    csv_logger_log_int(logger, find_col(logger, name), value);
}

void csv_logger_log_row(CSVLogger* logger, const double* values) {
    for (int i = 0; i < logger->num_cols; i++) {
        logger->row_values[i] = values[i];
        logger->row_set[i]    = true;
        logger->row_is_int[i] = false;
    }
}

void csv_logger_commit_row(CSVLogger* logger) {
    if (!logger->fp) return;
    for (int i = 0; i < logger->num_cols; i++) {
        if (i > 0) fputc(',', logger->fp);
        if (!logger->row_set[i]) {
            // empty cell
        } else if (logger->row_is_int[i]) {
            fprintf(logger->fp, "%d", (int)logger->row_values[i]);
        } else {
            double v = logger->row_values[i];
            // Use enough precision: if the value looks like an integer, skip trailing zeros
            if (v == floor(v) && fabs(v) < 1e15)
                fprintf(logger->fp, "%.1f", v);
            else
                fprintf(logger->fp, "%.7g", v);
        }
    }
    fputc('\n', logger->fp);

    // Reset pending row
    for (int i = 0; i < logger->num_cols; i++) {
        logger->row_values[i] = 0.0;
        logger->row_set[i]    = false;
        logger->row_is_int[i] = false;
    }

    logger->rows_since_flush++;
    if (logger->flush_interval <= 0 ||
        logger->rows_since_flush >= logger->flush_interval) {
        fflush(logger->fp);
        logger->rows_since_flush = 0;
    }
}

void csv_logger_flush(CSVLogger* logger) {
    if (logger->fp)
        fflush(logger->fp);
}

void csv_logger_close(CSVLogger* logger) {
    if (logger->fp) {
        fflush(logger->fp);
        fclose(logger->fp);
        logger->fp = NULL;
    }
    for (int i = 0; i < logger->num_cols; i++) {
        free(logger->col_names[i]);
        logger->col_names[i] = NULL;
    }
}
