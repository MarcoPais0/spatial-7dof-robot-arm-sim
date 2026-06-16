# 7DOF Robotic Arm DH Simulation

Spatial 7DOF robotic arm simulation with DH kinematics, Jacobian-based inverse differential kinematics, simplified joint dynamics, and a Qt-based dashboard visualization.

Supported Python versions for the demo venv are `3.13` and `3.12` as a fallback.
Create and run the app from the project venv, not from an arbitrary system Python.
The demo opens as a dark minimal dashboard window sized to the available display when it can,
instead of forcing fullscreen, and it relies on the OS title-bar controls for close/minimize.
If Qt, the display backend, or OpenGL support is missing, the demo stops early with a clear startup message instead of failing later.

## Quick Start

```bash
# See docs/environment.md for the exact Python install and venv setup commands.
./.venv/bin/python scripts/check_environment.py
./.venv/bin/python scripts/smoke_check.py
```

## Documentation

- [`docs/README.md`](docs/README.md) for the technical documentation index
- [`AGENTS.md`](AGENTS.md) for repo conventions and working notes
