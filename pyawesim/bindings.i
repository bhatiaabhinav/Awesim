/* bindings.i */

// Define the module name
%module bindings

// Enable Python 3 annotations using C types (for type hints)
%feature("python:annotations", "c");

// Include necessary system headers and your project headers (verbatim in wrapper C code)
%{
#include "../include/utils.h"
#include "../include/map.h"
#include "../include/car.h"
#include "../include/ai.h"
#include "../include/actions.h"
#include "../include/sim.h"
#include "../include/awesim.h"
%}

// Parse and wrap the headers
%include "../include/utils.h"
%include "../include/map.h"
%include "../include/car.h"
%include "../include/ai.h"
%include "../include/actions.h"
%include "../include/sim.h"
%include "../include/awesim.h"