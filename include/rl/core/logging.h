#ifndef logfGING_H
#define logfGING_H

#include <stdio.h>
#include <stdbool.h>

// ===========================================================================
// CSV Logger — column-aware CSV writer with periodic flushing
//
// Usage:
//   const char* cols[] = {"iter", "reward", "loss"};
//   CSVLogger lg = csv_logger("out.csv", 10, 3, cols);
//   csv_logger_log_int(&lg, 0, 1);          // by column index
//   csv_logger_log(&lg, 1, 3.14);           // by column index
//   csv_logger_log_by_name(&lg, "loss", 0.01);  // by name
//   csv_logger_commit_row(&lg);             // writes & resets row
//   csv_logger_close(&lg);
// ===========================================================================

#define CSV_LOGGER_MAX_COLS 64

typedef struct {
    FILE* fp;
    int   flush_interval;   // flush every N rows (0 = flush every row)
    int   rows_since_flush;
    // Column schema
    int   num_cols;
    char* col_names[CSV_LOGGER_MAX_COLS];
    // Pending row buffer
    double row_values[CSV_LOGGER_MAX_COLS];
    bool   row_set[CSV_LOGGER_MAX_COLS];    // which columns have been set
    bool   row_is_int[CSV_LOGGER_MAX_COLS]; // format as integer
} CSVLogger;

// Create a CSV logger writing to `filename`.
// Writes header immediately. Creates intermediate directories.
// flush_interval: flush every N rows (0 = flush every row).
CSVLogger csv_logger(const char* filename, int flush_interval,
                     int num_cols, const char* col_names[]);

// Set a column value by index (double).
void csv_logger_log(CSVLogger* logger, int col, double value);

// Set a column value by name (double).
void csv_logger_log_by_name(CSVLogger* logger, const char* name, double value);

// Set a column value by index (integer).
void csv_logger_log_int(CSVLogger* logger, int col, int value);

// Set a column value by name (integer).
void csv_logger_log_int_by_name(CSVLogger* logger, const char* name, int value);

// Set all column values at once (array of doubles, length must == num_cols).
void csv_logger_log_row(CSVLogger* logger, const double* values);

// Write the pending row to file and reset for the next row.
void csv_logger_commit_row(CSVLogger* logger);

// Force flush buffered output to disk.
void csv_logger_flush(CSVLogger* logger);

// Close the logger file handle and free column name copies.
void csv_logger_close(CSVLogger* logger);

#endif // logfGING_H
