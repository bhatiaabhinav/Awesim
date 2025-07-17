#include <stdlib.h>
#include "procedures.h"
#include "logging.h"


// --- ADJUST SPEED procedure ---

ProcedureStatusCode procedure_adjust_speed_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {
    return PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED; // Placeholder for actual implementation
}

ProcedureStatusCode procedure_adjust_speed_step(Simulation* sim, Car* car, Procedure* procedure) {
    return PROCEDURE_STATUS_FAILED_REASON_UNINITIALIZED; // Placeholder for actual implementation
}

void procedure_adjust_speed_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    return; // Placeholder for actual implementation
}