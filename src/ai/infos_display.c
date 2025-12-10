#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>


InfosDisplay* infos_display_malloc(int width, int height, int num_infos) {
    InfosDisplay* display = (InfosDisplay*)malloc(sizeof(InfosDisplay));
    if (!display) {
        LOG_ERROR("Failed to allocate InfosDisplay");
        return NULL;
    }
    display->width = width;
    display->height = height;
    display->num_infos = num_infos;

    display->data = (uint8_t*)malloc(3 * width * height);
    if (!display->data) {
        LOG_ERROR("Failed to allocate InfosDisplay data");
        free(display);
        return NULL;
    }
    display->info_data = (double*)malloc(sizeof(double) * num_infos);
    if (!display->info_data) {
        LOG_ERROR("Failed to allocate InfosDisplay info data");
        free(display->data);
        free(display);
        return NULL;
    }
    display->info_colors = (RGB*)malloc(sizeof(RGB) * num_infos);
    if (!display->info_colors) {
        LOG_ERROR("Failed to allocate InfosDisplay info colors");
        free(display->info_data);
        free(display->data);
        free(display);
        return NULL;
    }

    memset(display->info_data, 0, sizeof(double) * num_infos);
    memset(display->data, 0, 3 * width * height);
    memset(display->info_colors, 0xFF, sizeof(RGB) * num_infos); // default to white
    
    return display;
}

void infos_display_set_info(InfosDisplay* display, int info_index, double value) {
    if (!display) return;
    if (info_index < 0 || info_index >= display->num_infos) return;
    if (value < 0.0 || value > 1.0) {
        LOG_WARN("InfosDisplay: info value out of range [0.0, 1.0]: %f. Clamping.", value);
        value = (value < 0.0) ? 0.0 : 1.0;
    }
    display->info_data[info_index] = value;
}

void infos_display_set_info_color(InfosDisplay* display, int info_index, RGB color) {
    if (!display) return;
    if (info_index < 0 || info_index >= display->num_infos) return;
    display->info_colors[info_index] = color;
}

double infos_display_get_info(const InfosDisplay* display, int info_index) {
    if (!display) return 0.0;
    if (info_index < 0 || info_index >= display->num_infos) return 0.0;
    return display->info_data[info_index];
}

void infos_display_free(InfosDisplay* display) {
    if (display) {
        if (display->data) free(display->data);
        if (display->info_data) free(display->info_data);
        if (display->info_colors) free(display->info_colors);
        free(display);
    }
}

void infos_display_clear(InfosDisplay* display) {
    if (display) {
        for(int i=0; i<display->num_infos; i++) {
            display->info_data[i] = 0.0;
        }
    }
}

static void draw_bar(InfosDisplay* display, int x, int y, int w, int h, double level, RGB color) {
    double level_per_pixel = 1.0 / (double)w;
    for (int r = 0; r < h; r++) {
        int idx = ((y + r) * display->width + x) * 3; // Adjust idx for each row
        double level_remaining = level;
        for (int c = 0; c < w; c++) {
            // level remaining determines brightness. If level remaining covers more than half this pixel, full brightness. If less than half, dimmed brightness.
            double intensity = level_remaining >= level_per_pixel / 2 ? 1.0 : 0.2;

            display->data[idx++] = (uint8_t)(intensity * color.r); // R
            display->data[idx++] = (uint8_t)(intensity * color.g); // G
            display->data[idx++] = (uint8_t)(intensity * color.b); // B

            level_remaining -= level_per_pixel;
        }
    }
}

void infos_display_render(InfosDisplay* display) {
    if (!display || !display->data || !display->info_data) return;
    
    int num_cols = (int)(sqrt((double)display->num_infos));
    int num_rows = (display->num_infos + num_cols - 1) / num_cols;

    const int padding = 4; // from edges and between bars
    int remaining_width = display->width - (num_cols + 1) * padding;
    int remaining_height = display->height - (num_rows + 1) * padding;
    int bar_width = remaining_width / num_cols;
    int bar_height = remaining_height / num_rows;
    if (bar_height <= 0) bar_height = 1;
    if (bar_width <= 0) {
        LOG_WARN("InfosDisplay: insufficient width to render infos. Setting each bar 1 pixel wide, which can display only a binary level, where value >= 0.5 is full intensity.");
        return;
    }
    for (int i = 0; i < display->num_infos; i++) {
        int row = i / num_cols;
        int col = i % num_cols;
        int x = padding + col * (bar_width + padding);
        int y = padding + row * (bar_height + padding);
        double value = display->info_data[i];
        draw_bar(display, x, y, bar_width, bar_height, value, display->info_colors[i]);
    }
}
