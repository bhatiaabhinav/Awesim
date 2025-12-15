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
#include "../include/sim.h"
#include "../include/bad.h"
#include "../include/awesim.h"
%}

%include <stdint.i>  // SWIG typemap for uint32_t
// Parse and wrap the headers
%include "../include/utils.h"
%include "../include/map.h"
%include "../include/car.h"
%include "../include/ai.h"
%include "../include/sim.h"
%include "../include/bad.h"
%include "../include/awesim.h"

%extend RGBCamera {
    PyObject* _get_data_view() {
        if (!$self->data) {
            Py_RETURN_NONE;
        }
        Py_ssize_t size = (Py_ssize_t)($self->width * $self->height * 3);
        return PyMemoryView_FromMemory((char*)$self->data, size, PyBUF_WRITE);
    }
}

%extend InfosDisplay {
    PyObject* _get_data_view() {
        if (!$self->data) {
            Py_RETURN_NONE;
        }
        Py_ssize_t size = (Py_ssize_t)($self->width * $self->height * 3);
        return PyMemoryView_FromMemory((char*)$self->data, size, PyBUF_WRITE);
    }
}

%extend MiniMap {
    PyObject* _get_data_view() {
        if (!$self->data) {
            Py_RETURN_NONE;
        }
        Py_ssize_t size = (Py_ssize_t)($self->width * $self->height * 3);
        return PyMemoryView_FromMemory((char*)$self->data, size, PyBUF_WRITE);
    }
}

%pythoncode %{
    def _rgbcamera_to_numpy(self):
        """Returns the image data as a numpy array (H, W, 3).
           Note: The array shares memory with the C struct.
        """
        import numpy as np
        view = self._get_data_view()
        if view is None:
            return None
        return np.frombuffer(view, dtype=np.uint8).reshape((self.height, self.width, 3))

    RGBCamera.to_numpy = _rgbcamera_to_numpy
    InfosDisplay.to_numpy = _rgbcamera_to_numpy
    MiniMap.to_numpy = _rgbcamera_to_numpy
%}
