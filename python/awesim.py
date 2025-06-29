#  Before running this script, make sure you run compile_so.sh script on linux or mac, or compile_dll.bat on Windoes. Also, install `pip install cffi` in your python environment.

import os
from cffi import FFI

# os.environ["SIM_TRACE"] = "sim_logic.c"     # to print simulator logs

ffi = FFI()
try:
    with open(os.path.join("python", "libawesim.h")) as f:
        ffi.cdef(f.read(), override=True)
except FileNotFoundError:
    print("Error: libawesim.h not found")
    exit(1)

# Dynamic library loading based on OS.
lib_extension = "dll" if os.name == "nt" else "so"
lib_name = f"libawesim.{lib_extension}"
lib_path = os.path.abspath(os.path.join("bin", lib_name))
try:
    libsim = ffi.dlopen(lib_path)     # sim module. Contains everything declared in the header files
except OSError as e:
    print(f"Error loading library {lib_path}: {e}")
    exit(1)


sim = libsim.sim_malloc()
city_width = 1000   # 1km
num_cars = 256        # 8 cars (incl. agent)
init_clock = libsim.clock_reading(0, 8, 0, 0)   # Monday 8:00 AM
init_weather = libsim.WEATHER_SUNNY
libsim.awesim_setup(sim, city_width, num_cars, 0.02, init_clock, init_weather)  # set's up sim variables, awesim map and adds cars randomly
libsim.sim_set_agent_enabled(sim, True)         # car 0 won't be NPC
decision_interval = 0.1
total_play_time = 100.0
agent = libsim.sim_get_agent_car(sim)
libsim.sim_set_synchronized(sim, True, 1.0)  # synchronize simulation with real time
libsim.sim_connect_to_render_server(sim, "127.0.0.1".encode('utf-8'), 4242)  # before this, start the server with `./bin/awesim_render_server 4242 --persistent`

print(f"Running simulation for {total_play_time} seconds with decision interval of {decision_interval} seconds. Press Ctrl+C to stop.")
while libsim.sim_get_time(sim) < total_play_time:
    libsim.situational_awareness_build(sim, agent.id)  # holds most of the state variables needed for decision making
    situation = libsim.sim_get_situational_awareness(sim, agent.id)  # get situational awareness for the agent
    distance_to_lead_vehicle = situation.distance_to_lead_vehicle   # example variable
    # set action space variables:
    libsim.car_set_acceleration(agent, 4.0)     # 0-60mph in ~4.5s
    libsim.car_set_indicator_lane(agent, libsim.INDICATOR_NONE)
    libsim.car_set_indicator_turn(agent, libsim.INDICATOR_NONE)
    libsim.simulate(sim, decision_interval)    # simulate 0.1 seconds
print("Simulation finished.")

libsim.sim_disconnect_from_render_server(sim)  # close rendering window
libsim.sim_free(sim)
