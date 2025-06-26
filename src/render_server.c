#include "awesim.h"
#include "render.h"
#include "logging.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>

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

static int PORT = 4242;
static SimCommand command = COMMAND_NONE; // Current command from input
static int server_fd = -1; // Server socket file descriptor
static int client_fd = -1; // Client socket file descriptor
struct sockaddr_in server_addr = {0}; // Server address structure

// Initialize server socket (socket, bind, listen)
static bool init_server_socket() {
    #ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            LOG_ERROR("WSAStartup failed");
            return false;
        }
    #endif

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        LOG_ERROR("Socket creation failed");
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        LOG_ERROR("Bind failed");
        CLOSE_SOCKET(server_fd);
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    if (listen(server_fd, 1) == -1) {
        LOG_ERROR("Listen failed");
        CLOSE_SOCKET(server_fd);
        #ifdef _WIN32
            WSACleanup();
        #endif
        return false;
    }

    LOG_INFO("Listening on port %d...", PORT);
    return true;
}

// Accept a client connection
static bool accept_client() {
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
    if (client_fd == -1) {
        LOG_ERROR("Accept failed");
        return false;
    }
    LOG_INFO("Client connected");
    return true;
}

static bool send_command() {
    if (send(client_fd, (char*)&command, 1, 0) != 1) {
        LOG_ERROR("Failed to send command %u", command);
        return true;
    }
    if (command != COMMAND_NONE) {
        LOG_DEBUG("Command sent: %u", command);
    } else {
        LOG_TRACE("Command sent: %u", command);
    }
    command = COMMAND_NONE; // Reset command after sending
    return false; // Success
}

// Receive specified number of bytes using blocking socket
static int recv_full(int sockfd, char* buffer, int len) {
    int total_received = 0;
    while (total_received < len) {
        int bytes = recv(sockfd, buffer + total_received, len - total_received, 0);
        if (bytes <= 0) {
            if (bytes == 0) {
                LOG_DEBUG("Socket disconnected");
            } else {
                LOG_ERROR("Socket disconnected ungracefully or error occurred");
            }
            return -1; // Error or disconnection
        }
        total_received += bytes;
    }
    return total_received;
}

// Receive message using blocking sockets
static bool receive_message(Simulation* sim, char* receive_buffer) {
    uint32_t msg_len;
    if (recv_full(client_fd, (char*)&msg_len, sizeof(msg_len)) != sizeof(msg_len)) {
        return true; // Fatal error or disconnection
    }
    msg_len = ntohl(msg_len);
    if (msg_len > sizeof(Simulation)) {
        LOG_ERROR("Message too large (%u bytes)", msg_len);
        return true; // Fatal error
    }
    if (recv_full(client_fd, receive_buffer, msg_len) != msg_len) {
        return true; // Fatal error or disconnection
    }
    memcpy(sim, receive_buffer, msg_len);
    LOG_TRACE("Sim state received and copied");
    return false; // Success
}

int main(int argc, char* argv[]) {
    LOG_INFO("Starting render server");
    int port = PORT; // Default port
    bool persistent = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--persistent") == 0) {
            LOG_DEBUG("Persistent mode enabled");
            persistent = true;
        } else {
            port = atoi(argv[i]);
            if (port <= 0 || port > 65535) {
                LOG_ERROR("Invalid port number: %s. Using default port %d", argv[i], PORT);
                port = PORT;
            } else {
                PORT = port; // Update port if valid
            }
        }
    }
    Simulation* sim = sim_malloc();
    int buffer_size = sizeof(Simulation);
    char* receive_buffer = malloc(buffer_size);

    if (!sim || !receive_buffer) {
        LOG_ERROR("Memory allocation failed");
        sim_free(sim);
        free(receive_buffer);
        return EXIT_FAILURE;
    }

    // Initialize server socket
    if (!init_server_socket()) {
        sim_free(sim);
        free(receive_buffer);
        return EXIT_FAILURE;
    }

    // Initialize SDL
    if (!init_sdl()) {
        CLOSE_SOCKET(server_fd);
        #ifdef _WIN32
            WSACleanup();
        #endif
        sim_free(sim);
        free(receive_buffer);
        return EXIT_FAILURE;
    }

    bool quit = false;
    if (persistent) {
        while (!quit) {
            if (!accept_client()) {
                continue; // Try accepting another client
            }

            bool client_disconnected = false;
            while (!client_disconnected && !quit) {
                command = handle_sdl_events();
                if (command == COMMAND_QUIT) {
                    LOG_INFO("Quit event received.");
                    send_command();
                    quit = true;
                    break;
                }
                if (receive_message(sim, receive_buffer)) {
                    client_disconnected = true;
                    continue;
                }
                render(sim);
                if (send_command()) {
                    client_disconnected = true;
                }
            }

            CLOSE_SOCKET(client_fd);
            LOG_INFO("Client disconnected");
        }
    } else {
        if (!accept_client()) {
            CLOSE_SOCKET(server_fd);
            #ifdef _WIN32
                WSACleanup();
            #endif
            sim_free(sim);
            free(receive_buffer);
            return EXIT_FAILURE;
        }

        while (!quit) {
            command = handle_sdl_events();
            if (command == COMMAND_QUIT) {
                LOG_INFO("Quit event received.");
                quit = true;
                send_command();
                continue;
            }
            if (receive_message(sim, receive_buffer)) {
                quit = true;
                continue;
            }
            render(sim);
            send_command();
        }

        CLOSE_SOCKET(client_fd);
        LOG_INFO("Client disconnected");
    }

    // Cleanup
    LOG_DEBUG("Cleaning up resources");
    CLOSE_SOCKET(server_fd);
    #ifdef _WIN32
        WSACleanup();
    #endif
    cleanup_sdl();
    sim_free(sim);
    free(receive_buffer);
    LOG_INFO("Render server closed");
    return 0;
}