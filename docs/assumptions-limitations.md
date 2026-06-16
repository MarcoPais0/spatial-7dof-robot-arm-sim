# Assumptions and Limitations

## Intentional Simplifications

- The dynamics model is reduced to diagonal inertia, viscous damping, and gravity.
- Coriolis and centrifugal coupling terms are omitted.
- The controller is joint-space and does not solve a full orientation-tracking task.
- The inverse-kinematics stage prioritizes tool-position tracking.
- The current gravity torque model is the intended final v1 model, even though it remains intentionally lightweight.
- The arm geometry is a generic 7DOF industrial-style chain, not a manufacturer-accurate KUKA replication.
- The visualization shell is Qt Widgets plus pyqtgraph; the 3D card uses a lightweight Qt OpenGL viewport rather than a full graphics engine.
- The validated runtime is Python `3.13`, with Python `3.12` as fallback. Python `3.14` is not treated as a supported baseline for this repo because the GUI/OpenGL package layout has been unreliable there.

## Modeling Tradeoffs

- The arm is designed to be spatial rather than planar, but it is still a compact teaching model.
- The tool frame is aligned with `z7` so joint 7 acts as the roll DOF for the tool origin.
- The dashboard makes the seven joint origins explicit with markers and a `7 DOF` badge, but the visual chain still follows the FK frames rather than a stylized segmented mesh.
- The live charts use rounded 5-tick y-axis ladders that rescale automatically rather than a fully continuous auto-ranging plot.
- Singularity handling uses damping rather than constraint-based optimization.
- Numerical integration uses a fixed-step semi-implicit Euler method instead of a higher-order solver.

## What Is Not Modeled

- Full rigid-body coupling across joints
- Friction models beyond simple viscous damping
- Compliance, backlash, and actuator electronics
- Real hardware limits beyond basic saturation behavior
- A browser-style or Qt Quick dashboard layer

## Report Use

This document is the right place to capture what is intentionally abstracted away so the future academic report can distinguish model choices from physical reality.
