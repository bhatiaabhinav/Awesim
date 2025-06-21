#include "utils.h"
#include "render.h"

static double last_render_t = 0.0;  // Last render time for FPS calculation
static double render_fps = 0.0;     // Render FPS

void render(Simulation *sim) {
    bool draw_lanes = true;
    bool draw_cars = true;
    bool draw_track_lines = false;
    bool draw_traffic_lights = true;
    bool draw_car_ids = true;
    bool draw_lane_ids = true;
    bool draw_road_names = true;
    int hud_font_size = 20;
    bool benchmark = false;
    render_sim(renderer, sim, draw_lanes, draw_cars, draw_track_lines, draw_traffic_lights,
                draw_car_ids, draw_lane_ids, draw_road_names, hud_font_size, benchmark);

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