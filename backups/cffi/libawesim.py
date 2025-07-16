# Before running this script, make sure you run compile_so.sh script on linux or mac, or compile_dll.bat on Windows. Also, install `pip install cffi` in your python environment.

import os
import sys
import re
from cffi import FFI

# os.environ["SIM_TRACE"] = "sim_logic.c"     # to print simulator logs

ffi = FFI()
header_path = os.path.join("python", "libawesim.h")
try:
    with open(header_path) as f:
        header_content = f.read()
        ffi.cdef(header_content, override=True)
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

successful_functions = []
successful_constants = []

# Import all libsim attributes into the global namespace
for attr_name in dir(libsim):
    if not attr_name.startswith('__'):  # Skip internal attributes
        try:
            value = getattr(libsim, attr_name)
            globals()[attr_name] = value
            if callable(value):
                successful_functions.append(attr_name)
            else:
                successful_constants.append(attr_name)
        except NotImplementedError as e:
            print(f"\033[91mNotImplementedError while importing {attr_name}: {e}\033[0m")
        except AttributeError as e:
            print(f"\033[91mAttributeError while importing {attr_name}: {e}\033[0m")
        except Exception as e:
            print(f"\033[91mUnexpectedError while importing {attr_name}: {e}\033[0m")
