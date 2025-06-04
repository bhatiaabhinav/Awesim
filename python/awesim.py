import os
from cffi import FFI

os.environ["SIM_TRACE"] = "sim_logic.c"     # to print simulator logs

ffi = FFI()
with open(os.path.join("python", "libawesim.h")) as f:
    ffi.cdef(f.read(), override=True)
libsim = ffi.dlopen(os.path.join("bin", "libawesim.so"))   # sim module. Contains everything declared in the header files

# libsim.main(0, [])

sim = libsim.sim_malloc()
city_width = 1000   # 1km
num_cars = 2        # 8 cars (incl. agent)
init_clock = libsim.clock_reading(0, 8, 0, 0)   # Monday 8:00 AM
init_weather = libsim.WEATHER_SUNNY
libsim.awesim_setup(sim, city_width, num_cars, 0.02, init_clock, init_weather)  # set's up sim variables, awesim map and adds cars randomly
libsim.sim_set_agent_enabled(sim, True)         # car 0 won't be NPC
decision_frequency = 0.1
total_play_time = 8.0
agent = libsim.sim_get_agent_car(sim)

while libsim.sim_get_time(sim) < total_play_time:
    situation = libsim.situational_awareness_build(agent, sim)  # holds most variables needed for decision making
    distance_to_lead_vehicle = situation.distance_to_lead_vehicle
    libsim.car_set_acceleration(agent, 4.0)     # 0-60mph in ~4.5s
    libsim.car_set_indicator_lane(agent, libsim.INDICATOR_NONE)
    libsim.car_set_indicator_turn(agent, libsim.INDICATOR_NONE)
    libsim.simulate(sim, decision_frequency)    # simulate 0.1 seconds

libsim.sim_free(sim)
