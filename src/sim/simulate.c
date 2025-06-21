#include "sim.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #define CLOSE_SOCKET closesocket
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define CLOSE_SOCKET close
#endif

bool sim_connect_to_render_server(Simulation* self, const char* server_ip, int port) {
    if (self->is_connected_to_render_server) {
        LOG_WARN("Already connected to render server");
        return true;
    }

    #ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            LOG_ERROR("WSAStartup failed");
            return false;
        }
    #endif

    self->render_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (self->render_socket == -1) {
        LOG_ERROR("Socket creation failed");
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        LOG_ERROR("Invalid address");
        CLOSE_SOCKET(self->render_socket);
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    if (connect(self->render_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        LOG_ERROR("Connection to render server failed");
        CLOSE_SOCKET(self->render_socket);
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    self->is_connected_to_render_server = true;
    LOG_INFO("Connected to render server at %s:%d", server_ip, port);
    return true;
}

void sim_disconnect_from_render_server(Simulation* self) {
    if (!self->is_connected_to_render_server) {
        return;
    }
    CLOSE_SOCKET(self->render_socket);
    self->is_connected_to_render_server = false;
    #ifdef _WIN32
        WSACleanup();
    #endif
    LOG_INFO("Disconnected from render server");
}

static void sim_sync_with_render_server(Simulation* self) {
    if (!self || !self->is_connected_to_render_server) {
        LOG_ERROR("Simulation is NULL or not connected to render server");
        return;
    }
    uint32_t msg_len = sizeof(Simulation);
    uint32_t net_len = htonl(msg_len);
    if (send(self->render_socket, (char*)&net_len, sizeof(net_len), 0) != sizeof(net_len)) {
        LOG_ERROR("Failed to send message length");
        sim_disconnect_from_render_server(self);
        return;
    }
    if (send(self->render_socket, (char*)self, msg_len, 0) != msg_len) {
        LOG_ERROR("Failed to send simulation state");
        sim_disconnect_from_render_server(self);
        return;
    }
    LOG_TRACE("Sent simulation state to render server");

    SimCommand command = COMMAND_NONE;
    int bytes = recv(self->render_socket, (char*)&command, 1, 0);
    if (bytes == 1) {
        switch (command) {
            case COMMAND_SIM_DECREASE_SPEED: // Slow down
                self->simulation_speedup -= self->simulation_speedup < 1.01 ? 0.1 : 1.0;
                self->simulation_speedup = fmax(self->simulation_speedup, 0.0);
                LOG_DEBUG("Received slow_down command, new speedup: %.1f", self->simulation_speedup);
                break;
            case COMMAND_SIM_INCREASE_SPEED: // Speed up
                self->simulation_speedup += self->simulation_speedup < 0.99 ? 0.1 : 1.0;
                self->simulation_speedup = fmin(self->simulation_speedup, 100.0);
                LOG_DEBUG("Received speed_up command, new speedup: %.1f", self->simulation_speedup);
                break;
            case COMMAND_QUIT: // Quit
                if (self->should_quit_when_rendering_window_closed) {
                    LOG_DEBUG("Received quit command, stopping simulation");
                    sim_disconnect_from_render_server(self);
                    return;
                }
                break;
            case COMMAND_NONE: // Continue
                LOG_TRACE("Received continue command, continuing simulation");
                break;
            default:
                LOG_WARN("Invalid command received: %u", command);
                break;
        }
    } else {
        if (bytes == 0) {
            LOG_INFO("Render server disconnected");
        } else {
            LOG_ERROR("Failed to receive command");
        }
        sim_disconnect_from_render_server(self);
    }
}

void simulate(Simulation* self, Seconds sim_duration) {
    if (!self) {
        LOG_ERROR("Attempted to simulate a NULL Simulation pointer");
        return;
    }
    if (sim_duration <= 0) {
        LOG_ERROR("Invalid simulation duration: %.2f. It must be positive.", sim_duration);
        return;
    }
    if (!self->is_synchronized) {
        LOG_TRACE("Simulation is not synchronized with wall time, running as fast as possible.");
        sim_integrate(self, sim_duration);
        return; // No synchronization needed, just integrate
    }

    double sim_time = self->time;               // sim time passed in seconds
    double initial_sim_time = sim_time;         // store initial sim time
    double t_prev = get_sys_time_seconds();     // previous wall time in seconds
    LOG_TRACE("Starting simulation for %.2f seconds. Current sim time: %.2f seconds", 
              sim_duration, sim_time);

    while (sim_time - initial_sim_time < sim_duration) {
        if (!self->is_connected_to_render_server && self->should_quit_when_rendering_window_closed) {
            LOG_INFO("Render server disconnected, so, quitting simulation.");
            break;
        }

        double delta_wall_t = get_sys_time_seconds() - t_prev;
        double simulation_speedup = self->simulation_speedup;
        sim_time += delta_wall_t * simulation_speedup;
        sim_integrate(self, sim_time - self->time);     // make sim catch up to sim_time.
        t_prev = get_sys_time_seconds();                // update previous wall time

        // Send state to render server if connected
        if (self->is_connected_to_render_server) {
            sim_sync_with_render_server(self);  // If vsync is enabled, 60 FPS (or 120 FPS on some screens) is already enforced by the render server.
        } else {
            sleep_ms(10); // A little sleep allows for Ctrl+C to work in terminal.
        }

        LOG_TRACE("Simulation advanced to %.2f seconds (wall time delta: %.2f seconds)", sim_time, delta_wall_t);
    }
}