# Beck Stuff

Stuff for Doug Beck. I am back.

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and reproducibility. Follow these steps to set up and run the interactive GUI:

### 1. Install uv

If you haven’t already installed `uv`, you can follow the instructions here (https://docs.astral.sh/uv/getting-started/installation/):

### 2. Install QPV in Editable Mode

From the root directory (where `pyproject.toml` is located), run:

```
uv pip install -e .
```

This installs the QPV project. 

> You may sometimes have to trouble shoot and rm uv.lock, if you use uv sync, you will need to reinstall the project until I figure out how to stop it from being autoremoved. 

### 3. Launch the Interactive GUI

Still in the root directory, start the GUI with:

```
uv run src/qpv/position/interactive_gui.py
```

This command launches a Tk-based graphical interface for configuring verifiers, manipulating parameters, and visualizing the minimax intersection of light-speed spheres in 1D, 2D, or 3D.

> You must run this command from the directory containing `pyproject.toml` for everything to work correctly.

---

