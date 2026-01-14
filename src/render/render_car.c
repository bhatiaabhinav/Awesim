#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

SDL_Texture* car_id_texture_cache[MAX_CARS_IN_SIMULATION][MAX_FONT_SIZE] = {{NULL}};
SDL_Texture* car_speed_texture_cache[300][MAX_FONT_SIZE] = {{NULL}};
NearbyVehiclesFlattened HIGHLIGHTED_NEARBY_VEHICLES;
bool HIGHLIGHTED_CAR_AEB_ENGAGED;

static bool should_highlight_car(const Car* car) {
    // Check if the car ID is in the highlighted cars array
    for (int i = 0; HIGHLIGHTED_CARS[i] != ID_NULL; i++) {
        if (HIGHLIGHTED_CARS[i] == car->id) {
            return true;
        }
    }
    return false;
}

static bool should_highlight_as_a_nearby_vehicle(const Car* car) {
    // Check if it in HIGHLIGHTED_NEARBY_VEHICLES.car_ids array
    for (int i = 0; i < HIGHLIGHTED_NEARBY_VEHICLES.count; i++) {
        if (HIGHLIGHTED_NEARBY_VEHICLES.car_ids[i] == car->id) {
            return true;
        }
    }
    return false;
}

static bool should_highlight_as_a_forward_vehicle_when_aeb_engaged(const Car* car) {
    // Check if it is HIGHLIGHTED_NEARBY_VEHICLES.car_ids[1] since index 1 is the forward vehicle
    if (HIGHLIGHTED_NEARBY_VEHICLES.car_ids[1] == car->id) {
        return HIGHLIGHTED_CAR_AEB_ENGAGED; // return true if AEB is engaged
    }
    return false;
}

static void render_indicator(SDL_Renderer* renderer, const Car* car, CarIndicator indicator, double length, double width, int screen_width, int screen_height, int light_thickness, double cos_a, double sin_a, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {

    if (indicator == INDICATOR_NONE) return; // No indicator to render

    // Blinking logic: 500ms on, 500ms off (1000ms period)
    Uint32 current_time = SDL_GetTicks();
    const Uint32 blink_period = 1000; // 1 second
    const Uint32 blink_on_duration = 500; // 0.5 seconds

    if ((current_time % blink_period) > blink_on_duration) return;
    
    double indicator_length = width / 4; // Length of the indicator dash

    // Define local points for indicators
    Vec2D local_front_side_start, local_front_side_end; // Side indicator
    Vec2D local_back_start, local_back_end;             // Back indicator
    Vec2D local_front_edge_start, local_front_edge_end; // Original front edge indicator

    if (indicator == INDICATOR_RIGHT) { // Left side indicators
        // Front side indicator: left edge, 25% from front
        double x_pos = length / 2 - 0.25 * length; // 25% from front
        local_front_side_start = (Vec2D){x_pos - indicator_length / 2, -width / 2};
        local_front_side_end = (Vec2D){x_pos + indicator_length / 2, -width / 2};
        // Back indicator: left side of back edge
        local_back_start = (Vec2D){-length / 2, -width / 2};
        local_back_end = (Vec2D){-length / 2, -width / 2 + indicator_length};
        // Original front edge indicator: left side of front edge
        local_front_edge_start = (Vec2D){length / 2, -width / 2};
        local_front_edge_end = (Vec2D){length / 2, -width / 2 + indicator_length};
    } else if (indicator == INDICATOR_LEFT) { // INDICATOR_LEFT: Right side indicators
        // Front side indicator: right edge, 25% from front
        double x_pos = length / 2 - 0.25 * length; // 25% from front
        local_front_side_start = (Vec2D){x_pos - indicator_length / 2, width / 2};
        local_front_side_end = (Vec2D){x_pos + indicator_length / 2, width / 2};
        // Back indicator: right side of back edge
        local_back_start = (Vec2D){-length / 2, width / 2 - indicator_length};
        local_back_end = (Vec2D){-length / 2, width / 2};
        // Original front edge indicator: right side of front edge
        local_front_edge_start = (Vec2D){length / 2, width / 2 - indicator_length};
        local_front_edge_end = (Vec2D){length / 2, width / 2};
    } else {
        return; // Invalid indicator type
    }

    // Transform to world coordinates
    Coordinates world_front_side_start, world_front_side_end;
    Coordinates world_back_start, world_back_end;
    Coordinates world_front_edge_start, world_front_edge_end;

    // Front side start
    double x_local = local_front_side_start.x;
    double y_local = local_front_side_start.y;
    world_front_side_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_front_side_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Front side end
    x_local = local_front_side_end.x;
    y_local = local_front_side_end.y;
    world_front_side_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_front_side_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Back start
    x_local = local_back_start.x;
    y_local = local_back_start.y;
    world_back_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_back_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Back end
    x_local = local_back_end.x;
    y_local = local_back_end.y;
    world_back_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_back_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Original front edge start
    x_local = local_front_edge_start.x;
    y_local = local_front_edge_start.y;
    world_front_edge_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_front_edge_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Original front edge end
    x_local = local_front_edge_end.x;
    y_local = local_front_edge_end.y;
    world_front_edge_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
    world_front_edge_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

    // Convert to screen coordinates
    SDL_Point screen_front_side_start = to_screen_coords(world_front_side_start, screen_width, screen_height);
    SDL_Point screen_front_side_end = to_screen_coords(world_front_side_end, screen_width, screen_height);
    SDL_Point screen_back_start = to_screen_coords(world_back_start, screen_width, screen_height);
    SDL_Point screen_back_end = to_screen_coords(world_back_end, screen_width, screen_height);
    SDL_Point screen_front_edge_start = to_screen_coords(world_front_edge_start, screen_width, screen_height);
    SDL_Point screen_front_edge_end = to_screen_coords(world_front_edge_end, screen_width, screen_height);

    // Draw thick lines for indicators
    // FRONT_SIDE_INDICATOR
    thickLineRGBA_ignore_if_outside_screen(renderer, screen_front_side_start.x, screen_front_side_start.y, 
                                            screen_front_side_end.x, screen_front_side_end.y, light_thickness, r, g, b, a);
    // BACK_INDICATOR
    thickLineRGBA_ignore_if_outside_screen(renderer, screen_back_start.x, screen_back_start.y, 
                                            screen_back_end.x, screen_back_end.y, light_thickness, r, g, b, a);
    // FRONT_EDGE_INDICATOR (comment out if not needed)
    thickLineRGBA_ignore_if_outside_screen(renderer, screen_front_edge_start.x, screen_front_edge_start.y, 
                                            screen_front_edge_end.x, screen_front_edge_end.y, light_thickness, r, g, b, a);
}


void render_car(SDL_Renderer* renderer, const Car* car, Map* map, bool paint_id, bool paint_speed) {

    if (CAMERA_CENTERED_ON_CAR_ENABLED && car->id == CAMERA_CENTERED_ON_CAR_ID) {
        PAN_X = (int)(car->center.x * SCALE);
        PAN_Y = (int)(-car->center.y * SCALE);
    }

    double width = car->dimensions.x;
    double length = car->dimensions.y;
    double cos_a = cos(car->orientation);
    double sin_a = sin(car->orientation);

    // Convert to screen coordinates
    int screen_width = WINDOW_SIZE_WIDTH;
    int screen_height = WINDOW_SIZE_HEIGHT;
    SDL_Point screen_corners[4];
    for (int i = 0; i < 4; i++) {
        screen_corners[i] = to_screen_coords(car->corners[i], screen_width, screen_height);
    }

    // Draw car body
    Sint16 vx[4], vy[4];
    for (int i = 0; i < 4; i++) {
        vx[i] = (Sint16)screen_corners[i].x;
        vy[i] = (Sint16)screen_corners[i].y;
    }
    SDL_Color car_color = CAR_COLOR;
    if (should_highlight_car(car)) {
        car_color = HIGHLIGHTED_CAR_COLOR;
    } else if (should_highlight_as_a_nearby_vehicle(car)) {
        if (should_highlight_as_a_forward_vehicle_when_aeb_engaged(car)) {
            // Blinking logic: 100ms on, 100ms off (200ms period. Flash 5 times per second.)
            const Uint32 blink_period = 200; // 0.2 seconds
            const Uint32 blink_on_duration = 100; // 0.1 seconds
            Uint32 current_time = SDL_GetTicks();
            if ((current_time % blink_period) > blink_on_duration) {
                car_color = HIGHLIGHTED_FORWARD_VEHICLE_COLOR_AEB_ENGAGED;
            } else {
                car_color = HIGHLIGHTED_NEARBY_VEHICLES_COLOR;
            }
        } else {
            car_color = HIGHLIGHTED_NEARBY_VEHICLES_COLOR;
        }
    }
    filledPolygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 4, car_color.r, car_color.g, car_color.b, car_color.a);
    polygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 4, car_color.r, car_color.g, car_color.b, 255);

    int light_thickness = (int)(from_inches(6.0) * SCALE);
    light_thickness = fmax(light_thickness, 1); // Ensure light thickness is at least 1

    // Draw headlights (always on)
    {
        double headlight_length = width / 4 * 1.5; // 1.5x indicator length

        // Left headlight (front edge, left corner)
        Vec2D local_left_headlight_start = {length / 2, -width / 2};
        Vec2D local_left_headlight_end = {length / 2, -width / 2 + headlight_length};
        // Right headlight (front edge, right corner)
        Vec2D local_right_headlight_start = {length / 2, width / 2 - headlight_length};
        Vec2D local_right_headlight_end = {length / 2, width / 2};

        // Transform to world coordinates
        Coordinates world_left_headlight_start, world_left_headlight_end;
        Coordinates world_right_headlight_start, world_right_headlight_end;

        double x_local = local_left_headlight_start.x;
        double y_local = local_left_headlight_start.y;
        world_left_headlight_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_left_headlight_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_left_headlight_end.x;
        y_local = local_left_headlight_end.y;
        world_left_headlight_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_left_headlight_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_headlight_start.x;
        y_local = local_right_headlight_start.y;
        world_right_headlight_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_right_headlight_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_headlight_end.x;
        y_local = local_right_headlight_end.y;
        world_right_headlight_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_right_headlight_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

        // Convert to screen coordinates
        SDL_Point screen_left_headlight_start = to_screen_coords(world_left_headlight_start, screen_width, screen_height);
        SDL_Point screen_left_headlight_end = to_screen_coords(world_left_headlight_end, screen_width, screen_height);
        SDL_Point screen_right_headlight_start = to_screen_coords(world_right_headlight_start, screen_width, screen_height);
        SDL_Point screen_right_headlight_end = to_screen_coords(world_right_headlight_end, screen_width, screen_height);

        // Draw headlights (yellowish, 5px thick)
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_left_headlight_start.x, screen_left_headlight_start.y,
                                              screen_left_headlight_end.x, screen_left_headlight_end.y, light_thickness, 255, 255, 100, 255);
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_right_headlight_start.x, screen_right_headlight_start.y,
                                              screen_right_headlight_end.x, screen_right_headlight_end.y, light_thickness, 255, 255, 100, 255);
    }

    // Draw taillights
    bool is_braking = false;
    bool is_reversing = false;
    if (car->speed > 0 && car->acceleration < -0.01) {   // a little bit of tolerance since brake lights don't turn on with just a touch of brake
        is_braking = true; // Braking
    } else if (car->speed < -0 && car->acceleration > 0.01) {
        is_braking = true; // Reversing with brake
    } else if (car->speed < -0.001) {
        is_reversing = true; // reversing. Will render taillights in colors used for reversing in real cars, which is white-yellowish.
    }

    {
        double taillight_length = width / 4; // Same as indicator length

        // Left taillight
        Vec2D local_left_taillight_start = {-length / 2, -width / 2};
        Vec2D local_left_taillight_end = {-length / 2, -width / 2 + taillight_length};
        // Right taillight
        Vec2D local_right_taillight_start = {-length / 2, width / 2 - taillight_length};
        Vec2D local_right_taillight_end = {-length / 2, width / 2};

        // Transform to world coordinates
        Coordinates world_left_taillight_start, world_left_taillight_end;
        Coordinates world_right_taillight_start, world_right_taillight_end;

        double x_local = local_left_taillight_start.x;
        double y_local = local_left_taillight_start.y;
        world_left_taillight_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_left_taillight_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_left_taillight_end.x;
        y_local = local_left_taillight_end.y;
        world_left_taillight_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_left_taillight_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_taillight_start.x;
        y_local = local_right_taillight_start.y;
        world_right_taillight_start.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_right_taillight_start.y = car->center.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_taillight_end.x;
        y_local = local_right_taillight_end.y;
        world_right_taillight_end.x = car->center.x + x_local * cos_a - y_local * sin_a;
        world_right_taillight_end.y = car->center.y + x_local * sin_a + y_local * cos_a;

        // Convert to screen coordinates
        SDL_Point screen_left_taillight_start = to_screen_coords(world_left_taillight_start, screen_width, screen_height);
        SDL_Point screen_left_taillight_end = to_screen_coords(world_left_taillight_end, screen_width, screen_height);
        SDL_Point screen_right_taillight_start = to_screen_coords(world_right_taillight_start, screen_width, screen_height);
        SDL_Point screen_right_taillight_end = to_screen_coords(world_right_taillight_end, screen_width, screen_height);

        // Draw taillights (faint red normally, full red when braking)
        // Uint8 taillight_r = is_braking ? 255 : 100;
        // Uint8 taillight_g = 0;
        // Uint8 taillight_b = 0;

        Uint8 taillight_r, taillight_g, taillight_b;
        if (is_reversing) {
            taillight_r = 255; // White-yellowish for reversing
            taillight_g = 255;
            taillight_b = 128;
        } else if (is_braking) {
            taillight_r = 255; // Full red when braking
            taillight_g = 0;
            taillight_b = 0;
        } else {
            taillight_r = 100; // Faint red normally
            taillight_g = 0;
            taillight_b = 0;
        }

        Uint8 taillight_a = (is_braking || is_reversing) ? 255 : 150;
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_left_taillight_start.x, screen_left_taillight_start.y,
                                              screen_left_taillight_end.x, screen_left_taillight_end.y, light_thickness, taillight_r, taillight_g, taillight_b, taillight_a);
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_right_taillight_start.x, screen_right_taillight_start.y,
                                              screen_right_taillight_end.x, screen_right_taillight_end.y, light_thickness, taillight_r, taillight_g, taillight_b, taillight_a);
    }

    // render turn indicator with red, and lane change indicator with dark orange. turn indicator draws over lane change indicator.
    render_indicator(renderer, car, car->indicator_lane, length, width, screen_width, screen_height, light_thickness, cos_a, sin_a, 255, 140, 0, 255); // Lane change indicator in dark orange
    render_indicator(renderer, car, car->indicator_turn, length, width, screen_width, screen_height, light_thickness, cos_a, sin_a, 255, 0, 0, 255); // Turn indicator in red
    

    // Render car ID
    if (paint_id) {
        char id_str[10];
        snprintf(id_str, sizeof(id_str), "%d", car->id);
        int font_size = (int)(meters(1.0) * SCALE);     // font size = 1 meter
        SDL_Point car_center_screen = to_screen_coords( car->center, screen_width, screen_height);
        int text_x = car_center_screen.x;
        int text_y = car_center_screen.y;
        render_text(renderer, id_str, text_x, text_y, 255, 255, 255, 255, font_size, ALIGN_CENTER, false, car_id_texture_cache[car->id]);
    }

    // Render car speed
    if (paint_speed) {
        char speed_str[10];
        int speed_int = (int)to_mph(car->speed);
        snprintf(speed_str, sizeof(speed_str), "%d", speed_int);
        int font_size = (int)(meters(0.75) * SCALE);     // font size = 0.75 meter

        // offset in direction of the back of the car
        Vec2D speed_position = car->center;
        speed_position.x -= (length / 2 - 0.75) * cos_a; // 0.75 meters from the back of the car
        speed_position.y -= (length / 2 - 0.75) * sin_a; // 0.75 meters from the back of the car

        SDL_Point speed_position_screen = to_screen_coords(speed_position,  screen_width, screen_height);
        int text_x = speed_position_screen.x;
        int text_y = speed_position_screen.y;
        int cache_id = 100 + speed_int; // to avoid conflict with car ID cache (assuming speed ranged from -100 to 200 mph)
        render_text(renderer, speed_str, text_x, text_y, 255, 255, 255, 255, font_size, ALIGN_CENTER, false, car_speed_texture_cache[cache_id]);
    }
}