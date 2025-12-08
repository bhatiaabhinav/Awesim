#include "utils.h"
#include "render.h"

#define INV_60  0.01666666666666667
#define INV_120 0.00833333333333333
#define TARGET_INV_FPS INV_60

static double last_render_t = 0.0;  // Last render time for FPS calculation
static double render_fps = 0.0;     // Render FPS

void render(Simulation *sim) {
    render_sim(renderer, sim, DRAW_LANES, DRAW_CARS, DRAW_TRACK_LINES, DRAW_TRAFFIC_LIGHTS,
                DRAW_CAR_IDS, DRAW_CAR_SPEEDS, DRAW_LANE_IDS, DRAW_ROAD_NAMES, HUD_FONT_SIZE, false);

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

        // if vsync if off, and render time is less than 1/60 second, delay to cap at 60 FPS
        if (!VSYNC_ENABLED) {
            Seconds to_delay = -1;  // no delay by default
            if (render_time < TARGET_INV_FPS) to_delay = TARGET_INV_FPS - render_time;
            if (to_delay > 0) {
                SDL_Delay((Uint32)(to_delay * 1000));
                render_time = get_sys_time_seconds() - last_render_t;
            }
        }

        render_fps = 0.9 * render_fps + 0.1 / (render_time + 1e-6); // Exponential moving average of FPS
    }
    last_render_t = get_sys_time_seconds();
}