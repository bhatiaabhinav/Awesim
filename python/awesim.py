import os
from cffi import FFI

os.environ["SIM_TRACE"] = "sim_logic.c"     # to print simulator logs

ffi = FFI()
with open(os.path.join("python", "libawesim.h")) as f:
    ffi.cdef(f.read(), override=True)
libsim = ffi.dlopen(os.path.join("bin", "libawesim.so"))   # sim module. Contains everything declared in the header files

# sim.main(0, [])

sim = libsim.sim_malloc()
dt = 0.02
libsim.awesim_setup(sim, 256, 2, dt, libsim.clock_reading(0, 8, 0, 0), libsim.WEATHER_SUNNY)

libsim.simulate(sim, 5 * dt)  # simulate 0.1 seconds

libsim.sim_free(sim)
