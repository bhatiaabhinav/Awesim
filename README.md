<div style="text-align: center;">
  <img src="assets/icons/icon.png" alt="icon" width="200" height="200">
</div>

# Awesome 2D AV Simulator

Welcome to the Awesome 2D AV Simulator!

## Installation

First, ensure you have `gcc` and the SDL2 libraries installed. On Ubuntu, you can install the necessary dependencies with:

```bash
sudo apt install libsdl2-dev libsdl2-ttf-dev libsdl2-gfx-dev libsdl2-image-dev
```

On macOS, you can use Homebrew:

```bash
brew install sdl2 sdl2_ttf sdl2_gfx sdl2_image
```

On Windows, you would typically use minGW or a similar toolchain. Make sure to install the SDL2 libraries and set up your environment accordingly.

## Running the Simulator

To compile and launch the simulator, run:

```bash
sh scripts/linux/play.sh
```
For macOS, use `sh scripts/mac/play.sh`. For Windows, run `scripts/windows/play.bat`.

To benchmark the number of transitions per second:

```bash
sh scripts/linux/benchmark.sh
```
For macOS, use `sh scripts/mac/benchmark.sh`. For Windows, run `scripts/windows/benchmark.bat`.

## Exploring the Codebase

To understand how the simulator works, begin by reviewing the header files in the `include/` directory in the following order:

1. `utils.h`
2. `logging.h`
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

## Python Bindings

To generate bindings, first install swig using `sudo apt install swig` on Linux or `brew install swig` on macOS. On Windows, download the swig installer from the [SWIG website](http://www.swig.org/download.html) or use a package manager like Chocolatey or use [MSYS2](https://www.msys2.org/).

Create a Python virtual environment and install the required dependencies with `pip install -r pyawesim/requirements.txt`. Activate the virtual environment.

Finally, run `sh pyawesim/generate_bindings.sh` (on Linux/macOS) or `pyawesim\generate_bindings.bat` (on Windows) to generate the Python bindings.

See the `pyawesim/example.py` file for an example of how to use the bindings.

## Todos


## Contributing

Feel free to contribute or report issues. Enjoy simulating!