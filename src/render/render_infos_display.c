#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

// Render at bottom left corner of the screen
void render_infos_display(SDL_Renderer* renderer, const InfosDisplay* display) {
    if (!display || !display->data) return;

    // Create surface from pixel data
    // HWC format: Row-major, then column, then channel.
    // Pitch is width * 3 bytes.
    
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
        (void*)display->data,
        display->width,
        display->height,
        24,                 // depth
        display->width * 3, // pitch
        rmask, gmask, bmask, amask
    );

    if (!surface) {
        LOG_ERROR("Failed to create surface for infos display render: %s", SDL_GetError());
        return;
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface); // Texture has a copy now

    if (!texture) {
        LOG_ERROR("Failed to create texture for infos display render: %s", SDL_GetError());
        return;
    }

    // Render to bottom left
    SDL_Rect dest_rect;
    dest_rect.w = display->width;
    dest_rect.h = display->height;
    dest_rect.x = 20; // 20px padding from left
    dest_rect.y = WINDOW_SIZE_HEIGHT - display->height - 20; // 20px padding from bottom

    // Draw a border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White border
    SDL_Rect border_rect = { dest_rect.x - 2, dest_rect.y - 2, dest_rect.w + 4, dest_rect.h + 4 };
    SDL_RenderDrawRect(renderer, &border_rect);

    SDL_RenderCopy(renderer, texture, NULL, &dest_rect);
    SDL_DestroyTexture(texture);
}
