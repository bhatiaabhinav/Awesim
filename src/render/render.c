#include "utils.h"
#include "render.h"

static double last_render_t = 0.0;  // Last render time for FPS calculation
static double render_fps = 0.0;     // Render FPS

void render(Simulation *sim) {
    render_sim(renderer, sim, DRAW_LANES, DRAW_CARS, DRAW_TRACK_LINES, DRAW_TRAFFIC_LIGHTS,
                DRAW_CAR_IDS, DRAW_CAR_SPEEDS, DRAW_LANE_IDS, DRAW_ROAD_NAMES, DRAW_LIDAR, DRAW_CAMERA, DRAW_MINIMAP, DRAW_INFOS_DISPLAY, HUD_FONT_SIZE, false);

    // Render FPS stats
    char fps_stats[24];
    snprintf(fps_stats, sizeof(fps_stats), "FPS: %d   (VSync %s)", (int)render_fps, VSYNC_ENABLED ? "On" : "Off");
    int text_font_size = 20; // Default font size
    render_text(renderer, fps_stats, 10, 10, 255, 255, 255, 255, text_font_size,
                ALIGN_TOP_LEFT, false, NULL);

    SDL_RenderPresent(renderer);

    if (last_render_t > 0) {
        double render_time = get_sys_time_seconds() - last_render_t;
        render_time = render_time > 0 ? render_time : 0;    // clamp to 0 if negative due to clock issues

        render_fps = 0.95 * render_fps + 0.05 / (render_time + 1e-6); // Exponential moving average of FPS
    }
    last_render_t = get_sys_time_seconds();
}