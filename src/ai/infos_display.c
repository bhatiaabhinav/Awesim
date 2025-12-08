#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>

// Helper to draw a full intensity bar into a buffer
static void draw_static_bar(uint8_t* buffer, int width, int height, int x, int y, int w, int h, RGB color) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int px = x + c;
            int py = y + r;
            if (px >= width || py >= height) continue;
            
            int idx = (py * width + px) * 3;
            buffer[idx+0] = color.r;
            buffer[idx+1] = color.g;
            buffer[idx+2] = color.b;
        }
    }
}

// Initialize the static layout
static void infos_display_init_static(InfosDisplay* display) {
    if (!display || !display->static_data) return;
    
    memset(display->static_data, 0, 3 * display->width * display->height);
    
    int margin = 5;
    int col_gap = 10;
    int row_gap = 5;
    
    int total_w = display->width - 2 * margin - 2 * col_gap;
    int col_w = total_w / 3;
    
    int num_rows = display->num_rows;
    if (num_rows < 1) num_rows = 1;
    if (num_rows > MAX_INFOS_ROWS) num_rows = MAX_INFOS_ROWS;

    int total_h = display->height - 2 * margin - (num_rows - 1) * row_gap;
    int row_h = total_h / num_rows;
    
    if (row_h < 1) row_h = 1;
    
    // Colors (24 distinct colors)
    RGB colors_left[8] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {0, 255, 255}, {255, 0, 255}, {192, 192, 192}, {128, 0, 0}
    };
    RGB colors_middle[8] = {
        {128, 128, 0}, {0, 128, 0}, {128, 0, 128}, {0, 128, 128},
        {0, 0, 128}, {255, 165, 0}, {255, 192, 203}, {75, 0, 130}
    };
    RGB colors_right[8] = {
        {240, 230, 140}, {255, 127, 80}, {220, 20, 60}, {0, 255, 127},
        {70, 130, 180}, {218, 112, 214}, {255, 20, 147}, {139, 69, 19}
    };
    
    for (int i = 0; i < num_rows; i++) {
        int y = margin + i * (row_h + row_gap);
        draw_static_bar(display->static_data, display->width, display->height, margin, y, col_w, row_h, colors_left[i]);
        draw_static_bar(display->static_data, display->width, display->height, margin + col_w + col_gap, y, col_w, row_h, colors_middle[i]);
        draw_static_bar(display->static_data, display->width, display->height, margin + 2 * (col_w + col_gap), y, col_w, row_h, colors_right[i]);
    }
}

InfosDisplay* infos_display_malloc(int width, int height, int num_rows) {
    InfosDisplay* display = (InfosDisplay*)malloc(sizeof(InfosDisplay));
    if (!display) {
        LOG_ERROR("Failed to allocate InfosDisplay");
        return NULL;
    }
    display->width = width;
    display->height = height;
    display->num_rows = (num_rows > MAX_INFOS_ROWS) ? MAX_INFOS_ROWS : ((num_rows < 1) ? 1 : num_rows);

    display->data = (uint8_t*)malloc(3 * width * height);
    if (!display->data) {
        LOG_ERROR("Failed to allocate InfosDisplay data");
        free(display);
        return NULL;
    }
    display->static_data = (uint8_t*)malloc(3 * width * height);
    if (!display->static_data) {
        LOG_ERROR("Failed to allocate InfosDisplay static data");
        free(display->data);
        free(display);
        return NULL;
    }
    
    memset(display->data, 0, 3 * width * height);
    infos_display_init_static(display);
    
    for(int i=0; i<MAX_INFOS_ROWS; i++) {
        display->left_info[i] = 0.0;
        display->middle_info[i] = 0.0;
        display->right_info[i] = 0.0;
    }
    return display;
}

void infos_display_reset_num_rows(InfosDisplay* display, int num_rows) {
    if (!display) return;
    display->num_rows = (num_rows > MAX_INFOS_ROWS) ? MAX_INFOS_ROWS : ((num_rows < 1) ? 1 : num_rows);
    infos_display_init_static(display);
}

void infos_display_free(InfosDisplay* display) {
    if (display) {
        if (display->data) free(display->data);
        if (display->static_data) free(display->static_data);
        free(display);
    }
}

void infos_display_clear(InfosDisplay* display) {
    if (display) {
        // No need to clear data buffer as it will be overwritten by memcpy in render
        for(int i=0; i<MAX_INFOS_ROWS; i++) {
            display->left_info[i] = 0.0;
            display->middle_info[i] = 0.0;
            display->right_info[i] = 0.0;
        }
    }
}

static void dim_bar_part(InfosDisplay* display, int x, int y, int w, int h, double level) {
    int start_col = (int)(w * level);
    if (start_col >= w) return; // Full bar, no dimming needed
    if (start_col < 0) start_col = 0;
    
    for (int r = 0; r < h; r++) {
        for (int c = start_col; c < w; c++) {
             int px = x + c;
             int py = y + r;
             int idx = (py * display->width + px) * 3;
             // Apply 0.2 factor
             display->data[idx+0] = (uint8_t)(display->data[idx+0] * 0.2);
             display->data[idx+1] = (uint8_t)(display->data[idx+1] * 0.2);
             display->data[idx+2] = (uint8_t)(display->data[idx+2] * 0.2);
        }
    }
}

void infos_display_render(InfosDisplay* display) {
    if (!display || !display->data || !display->static_data) return;
    
    // Copy static layout (full intensity)
    memcpy(display->data, display->static_data, 3 * display->width * display->height);
    
    int margin = 5;
    int col_gap = 10;
    int row_gap = 5;
    
    int total_w = display->width - 2 * margin - 2 * col_gap;
    int col_w = total_w / 3;
    
    int num_rows = display->num_rows;
    if (num_rows < 1) num_rows = 1;
    if (num_rows > MAX_INFOS_ROWS) num_rows = MAX_INFOS_ROWS;

    int total_h = display->height - 2 * margin - (num_rows - 1) * row_gap;
    int row_h = total_h / num_rows;
    
    if (row_h < 1) row_h = 1;

    for (int i = 0; i < num_rows; i++) {
        int y = margin + i * (row_h + row_gap);
        
        // Left
        dim_bar_part(display, margin, y, col_w, row_h, display->left_info[i]);
        
        // Middle
        dim_bar_part(display, margin + col_w + col_gap, y, col_w, row_h, display->middle_info[i]);
        
        // Right
        dim_bar_part(display, margin + 2 * (col_w + col_gap), y, col_w, row_h, display->right_info[i]);
    }
}
