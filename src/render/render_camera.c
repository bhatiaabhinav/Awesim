#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>


// Render at bottom right corner of the screen (of size WINDOW_SIZE_WIDTH x WINDOW_SIZE_HEIGHT)
void render_camera(SDL_Renderer* renderer, const RGBCamera* camera) {
    if (!camera || !camera->data) return;

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
        (void*)camera->data,
        camera->width,
        camera->height,
        24,                 // depth
        camera->width * 3,  // pitch
        rmask, gmask, bmask, amask
    );

    if (!surface) {
        LOG_ERROR("Failed to create surface for camera render: %s", SDL_GetError());
        return;
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface); // Texture has a copy now

    if (!texture) {
        LOG_ERROR("Failed to create texture for camera render: %s", SDL_GetError());
        return;
    }

    // Render to bottom center
    SDL_Rect dest_rect;
    dest_rect.w = camera->width;
    dest_rect.h = camera->height;
    dest_rect.x = (WINDOW_SIZE_WIDTH - camera->width) / 2;
    dest_rect.y = WINDOW_SIZE_HEIGHT - camera->height - 20; // 20px padding

    // Draw a border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White border
    SDL_Rect border_rect = { dest_rect.x - 2, dest_rect.y - 2, dest_rect.w + 4, dest_rect.h + 4 };
    SDL_RenderDrawRect(renderer, &border_rect);

    // if the camera has a type, draw it below the image
    if (camera->attached_car) {
        const char* type_str = "Unknown";
        switch (camera->attached_car_camera_type) {
            case CAR_CAMERA_MAIN_FORWARD: type_str = "Main Forward"; break;
            case CAR_CAMERA_WIDE_FORWARD: type_str = "Wide Forward"; break;
            case CAR_CAMERA_SIDE_LEFT_FORWARDVIEW: type_str = "Side Left Forward"; break;
            case CAR_CAMERA_SIDE_RIGHT_FORWARDVIEW: type_str = "Side Right Forward"; break;
            case CAR_CAMERA_SIDE_LEFT_REARVIEW: type_str = "Side Left Rear"; break;
            case CAR_CAMERA_SIDE_RIGHT_REARVIEW: type_str = "Side Right Rear"; break;
            case CAR_CAMERA_REARVIEW: type_str = "Rearview"; break;
            case CAR_CAMERA_FRONT_BUMPER: type_str = "Front Bumper"; break;
            case CAR_CAMERA_NARROW_FORWARD: type_str = "Narrow Forward"; break;
            default: type_str = "Unknown"; break;
        }
        
        // Draw text centered below the image
        // stringRGBA draws 8x8 characters
        int text_width = strlen(type_str) * 8;
        int text_x = dest_rect.x + (dest_rect.w - text_width) / 2;
        int text_y = dest_rect.y + dest_rect.h + 5; // 5px padding below image
        
        stringRGBA(renderer, text_x, text_y, type_str, 255, 255, 255, 255);
    }

    SDL_RenderCopy(renderer, texture, NULL, &dest_rect);
    SDL_DestroyTexture(texture);
}