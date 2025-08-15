/* bindings.i */

// Define the module name
%module bindings

// Enable Python 3 annotations using C types (for type hints)
%feature("python:annotations", "c");

// Custom typemaps for (const double* args, int num_args)
%typemap(in) (const double* args, int num_args) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Expected a sequence (list, tuple, or array)");
    SWIG_fail;
  }
  $2 = (int) PySequence_Length($input);  // Set num_args
  if ($2 == 0) {
    $1 = NULL;  // Handle empty sequence if allowed
  } else {
    $1 = (double*) malloc($2 * sizeof(double));
    if (!$1) {
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
      SWIG_fail;
    }
    for (int i = 0; i < $2; i++) {
      PyObject *o = PySequence_GetItem($input, i);
      if (PyNumber_Check(o)) {
        $1[i] = PyFloat_AsDouble(o);
        Py_DECREF(o);
      } else {
        Py_DECREF(o);
        free($1);
        PyErr_SetString(PyExc_ValueError, "Sequence elements must be numbers");
        SWIG_fail;
      }
    }
  }
}

%typemap(freearg) (const double* args, int num_args) {
  if ($1) free((void*)$1);
}

// Include necessary system headers and your project headers (verbatim in wrapper C code)
%{
#include "../include/utils.h"
#include "../include/map.h"
#include "../include/car.h"
#include "../include/ai.h"
#include "../include/actions.h"
#include "../include/procedures.h"
#include "../include/sim.h"
#include "../include/awesim.h"
%}

%include <stdint.i>  // SWIG typemap for uint32_t
// Parse and wrap the headers
%include "../include/utils.h"
%include "../include/map.h"
%include "../include/car.h"
%include "../include/ai.h"
%include "../include/actions.h"
%include "../include/procedures.h"
%include "../include/sim.h"
%include "../include/awesim.h"