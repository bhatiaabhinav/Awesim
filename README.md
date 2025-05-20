# Awesome 2D AV Simulator

## Installation

First, ensure you have `gcc` and the SDL2 libraries installed. On Ubuntu, you can install the necessary dependencies with:

```bash
sudo apt install libsdl2-dev libsdl2-ttf-dev libsdl2-gfx-dev
```

## Running the Simulator

To compile and launch the simulator, run:

```bash
sh play.sh
```

To benchmark the number of transitions per second:

```bash
sh benchmark.sh
```

## Exploring the Codebase

To understand how the simulator works, begin by reviewing the header files in the `include/` directory in the following order:

1. `utils.h`
2. `road.h`
3. `map.h`
4. `car.h`
5. `ai.h`
6. `sim.h`
7. `awesim.h`
8. `render.h`

Each header file is implemented in a corresponding folder within the `src/` directory. For example, the `render.h` file is implemented in `src/render/`, with source files like:

- `src/render/render_lane.c`
- `src/render/render_car.c`
- etc.

---

Feel free to contribute or report issues. Enjoy simulating!

## Todos
