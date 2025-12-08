#include "utils.h"
#include <stdlib.h>

bool object_array_contains_object(void** objects_array, const void* object) {
    if (objects_array == NULL) return false;
    for (int i = 0; objects_array[i] != NULL; i++) {
        if (objects_array[i] == object) {
            return true;
        }
    }
    return false;
}