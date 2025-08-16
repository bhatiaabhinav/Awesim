import numpy as np
import bindings as A


def direction_onehot(direction) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    if direction == A.DIRECTION_NORTH:
        onehot[0] = 1.0
    elif direction == A.DIRECTION_EAST:
        onehot[1] = 1.0
    elif direction == A.DIRECTION_SOUTH:
        onehot[2] = 1.0
    elif direction == A.DIRECTION_WEST:
        onehot[3] = 1.0
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return onehot


def indicator_onehot(indicator) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    if indicator == A.INDICATOR_LEFT:
        onehot[0] = 1.0
    elif indicator == A.INDICATOR_NONE:
        onehot[1] = 1.0
    elif indicator == A.INDICATOR_RIGHT:
        onehot[2] = 1.0
    else:
        raise ValueError(f"Unknown indicator: {indicator}")
    return onehot
