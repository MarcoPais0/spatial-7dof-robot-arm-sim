# Environment

## Purpose

This project is intended to run from a local virtual environment built with Python `3.13`, with Python `3.12` as the fallback if `3.13` is unavailable on the machine.
The GUI stack is constrained to a conservative set so the Qt dashboard starts with a known plugin layout and the OpenGL imports are explicit.

## Supported Runtime

- Python `3.13` preferred
- Python `3.12` fallback
- The demo should be launched from the repository venv, not from the system Python

## Setup

Install Python `3.13` first. If `3.13` is unavailable on your machine, use `3.12` instead.

```bash
# confirm the interpreter you want to use
python3.13 --version
# or, if Python is installed in the standard macOS framework location
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 --version
```

Create the repository virtual environment with that interpreter:

```bash
cd /Users/marco.cruz.pais/Documents/GitHub/4dof-robot-arm-sim
rm -rf .venv
python3.13 -m venv .venv
# if 3.13 is not installed, use python3.12 -m venv .venv instead
```

Activate it, install dependencies, and run the environment check:

```bash
source .venv/bin/activate
python --version
which python

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/smoke_check.py
python src/robot_arm_3d_demo.py --mode cartesian
python src/robot_arm_3d_demo.py --mode joint
```

If you prefer not to activate the venv, use the absolute venv interpreter path instead of `python`:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/check_environment.py
./.venv/bin/python scripts/smoke_check.py
./.venv/bin/python src/robot_arm_3d_demo.py --mode cartesian
./.venv/bin/python src/robot_arm_3d_demo.py --mode joint
```

## What The Check Verifies

- the interpreter version is one of the supported versions
- `numpy`, `PySide6`, `pyqtgraph`, `OpenGL`, `OpenGL.GL`, and `pyqtgraph.opengl` import correctly
- the Qt plugin directory resolved from `PySide6` exists
- the Cocoa platform plugin file `libqcocoa.dylib` exists under the Qt `platforms` directory

## Startup Behavior

- The dashboard sets `QT_PLUGIN_PATH` and `QT_QPA_PLATFORM_PLUGIN_PATH` from the `PySide6` installation before `QApplication` is created.
- If the environment is incomplete, startup now reports the exact missing component instead of a generic reinstall message.
- The demo expects a working desktop session with display and OpenGL support.
