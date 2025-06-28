#include "utils.h"
#include "render.h"

static double last_render_t = 0.0;  // Last render time for FPS calculation
static double render_fps = 0.0;     // Render FPS

void render(Simulation *sim) {
    render_sim(renderer, sim, DRAW_LANES, DRAW_CARS, DRAW_TRACK_LINES, DRAW_TRAFFIC_LIGHTS,
                DRAW_CAR_IDS, DRAW_LANE_IDS, DRAW_ROAD_NAMES, HUD_FONT_SIZE, false);

    // Render FPS stats
    char fps_stats[24];
    snprintf(fps_stats, sizeof(fps_stats), "FPS: %d   (VSync On)", (int)render_fps);
    int text_font_size = 20; // Default font size
    render_text(renderer, fps_stats, 10, 10, 255, 255, 255, 255, text_font_size,
                ALIGN_TOP_LEFT, false, NULL);

    SDL_RenderPresent(renderer);

    double current_time = get_sys_time_seconds();
    if (last_render_t > 0) {
        double render_time = current_time - last_render_t;
        render_fps = 0.9 * render_fps + 0.1 / render_time;
    }
    last_render_t = current_time;
}