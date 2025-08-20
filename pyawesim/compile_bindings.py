# compile_bindings.py
import sys
from setuptools import setup, Extension
import glob

# List all C source files using glob to match the wildcards in your GCC command
c_sources = (
    glob.glob('../src/utils/*.c') +
    glob.glob('../src/map/*.c') +
    glob.glob('../src/sim/*.c') +
    glob.glob('../src/awesim/*.c') +
    glob.glob('../src/car/*.c') +
    glob.glob('../src/ai/*.c') +
    glob.glob('../src/logging/*.c')
)

# The SWIG-generated wrapper
swig_source = ['bindings_wrap.c']

# Platform-specific settings
define_macros = []
libraries = []
extra_compile_args = ['-Wall']
if sys.platform == 'win32':
    define_macros = [('_WIN32_WINNT', '0x0A00')]
    libraries.append('ws2_32')
elif sys.platform.startswith('linux'):
    libraries.append('rt')
# macOS (darwin) uses defaults

# Define the extension module, compiling all sources directly
module = Extension(
    '_bindings',  # The underlying C module (imported by bindings.py)
    sources=swig_source + c_sources,
    include_dirs=['../include'],  # -Iinclude
    define_macros=define_macros,  # -D_WIN32_WINNT=0x0A00 # type: ignore
    libraries=libraries,  # -lm -lws2_32 (math and winsock2)
    extra_compile_args=extra_compile_args,  # -Wall -Wunused-variable
    # Note: -fPIC and -shared are handled automatically by setuptools for extensions
)

setup(
    name='bindings',
    version='1.0',
    ext_modules=[module],
    py_modules=['bindings'],  # The SWIG-generated Python file
)
