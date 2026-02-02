#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static const char* nav_action_strings[] = {
    [NAV_STRAIGHT] = "Continue",
    [NAV_TURN_LEFT] = "Turn Left",
    [NAV_KEEP_STRAIGHT] = "Keep Straight",
    [NAV_TURN_RIGHT] = "Turn Right",
    [NAV_LANE_CHANGE_LEFT] = "Lane Change Left",
    [NAV_LANE_CHANGE_RIGHT] = "Lane Change Right",
    [NAV_MERGE_LEFT] = "Merge Left",
    [NAV_MERGE_RIGHT] = "Merge Right",
    [NAV_EXIT_LEFT] = "Exit Left",
    [NAV_EXIT_RIGHT] = "Exit Right",
    [NAV_NONE] = "None"
};

// Render at top right corner of the screen
void render_minimap(SDL_Renderer* renderer, const MiniMap* minimap) {
    if (!minimap || !minimap->data) return;

    // Create surface from pixel data
    Uint32 rmask, gmask, bmask, amask;
    #if SDL_BYTEORDER == SDL_BIG_ENDIAN
        rmask = 0xff0000;
        gmask = 0x00ff00;
        bmask = 0x0000ff;
        amask = 0;
    #else
        rmask = 0x0000ff;
        gmask = 0x00ff00;
        bmask = 0xff0000;
        amask = 0;
    #endif

    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
        (void*)minimap->data,
        minimap->width,
        minimap->height,
        24,                 // depth
        minimap->width * 3, // pitch
        rmask, gmask, bmask, amask
    );

    if (!surface) {
        LOG_ERROR("Failed to create surface for minimap render: %s", SDL_GetError());
        return;
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);

    if (!texture) {
        LOG_ERROR("Failed to create texture for minimap render: %s", SDL_GetError());
        return;
    }

    // Render to bottom right
    SDL_Rect dest_rect;
    dest_rect.w = minimap->width;
    dest_rect.h = minimap->height;
    dest_rect.x = WINDOW_SIZE_WIDTH - minimap->width - 20; // 20px padding from right
    dest_rect.y = WINDOW_SIZE_HEIGHT - minimap->height - 20; // 20px padding from bottom

    // Draw a border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White border
    SDL_Rect border_rect = { dest_rect.x - 2, dest_rect.y - 2, dest_rect.w + 4, dest_rect.h + 4 };
    SDL_RenderDrawRect(renderer, &border_rect);

    // Draw text below minimap if path is marked
    if (minimap->marked_path && minimap->marked_path->path_exists) {
        PathPlanner* planner = minimap->marked_path;
        
        // Find first non-trivial action
        NavAction next_action = path_planner_get_solution_action(planner, true, 0);
        
        const char* action_str = nav_action_strings[next_action];
        
        char info_text[128];
        if (planner->consider_speed_limits) {
            // Cost is time in seconds
            double minutes = planner->optimal_cost / 60.0;
            snprintf(info_text, sizeof(info_text), "%s, ETA: %.1f min", action_str, minutes);
        } else {
            // Cost is distance in meters
            snprintf(info_text, sizeof(info_text), "%s, DTG: %.0f m", action_str, planner->optimal_cost);
        }
        
        // Draw text right-aligned below the minimap
        int text_width = strlen(info_text) * 8;
        int text_x = dest_rect.x + dest_rect.w - text_width;
        int text_y = dest_rect.y + dest_rect.h + 5; // 5px padding below minimap
        
        stringRGBA(renderer, text_x, text_y, info_text, 255, 255, 255, 255);
    }

    SDL_RenderCopy(renderer, texture, NULL, &dest_rect);
    SDL_DestroyTexture(texture);
}
