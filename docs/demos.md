# Demos

## Purpose

The demos show whether the 7DOF kinematic model, inverse kinematics, controller, and dynamics work together as intended.
Run them from the repository venv with `./.venv/bin/python src/robot_arm_3d_demo.py --mode cartesian` for the main demo or `--mode joint` for the validation demo.
Use the repository venv created with Python `3.13`, or `3.12` if `3.13` is unavailable. The startup check in [`environment.md`](environment.md) should pass before launching either demo.
Startup is validated before the window opens, so missing Qt packages, a missing display backend, or missing OpenGL support fail fast with a direct message.

## Cartesian Tracking Demo

This is the main demo for the project. It exercises the full task-space pipeline.
It opens as a dark, minimal desktop dashboard sized to the available display,
with the simulation as the dominant card on the right and a stack of separate
metric cards on the left, separated by visible gutters. The visual theme keeps a
graphite background, darker card panels, and high-contrast blue, red, green, and
orange accents. The simulation card now shows each joint origin with visible
markers and a `7 DOF` badge. Each metric card shows the chart title in its
header strip, a fixed hover info badge immediately before the title, and a
live value beside it. Tooltips open on click or after a short hover delay, use
a black background with white text, and show a technical line plus a plain-
language line. Titles stay on one line. Clicking a joint chip toggles the
dashboard into that joint's view, and clicking the same chip again returns to
the original per-demo graphs without clearing the buffered history. The left
metric column takes roughly 40% of the window width.
The `J1` through `J7` chips in the simulation header also show each joint's DH
parameters on hover, and clicking one of them switches all four charts to that
joint's angle, velocity, torque, and power.
Use the normal OS title-bar controls to close it, or press `Esc` or `q` as a
keyboard fallback.
Resolved-rate IK generates a moving joint reference, and the joint-space
controller plus simplified dynamics track that reference while the target
sequence advances.

The new arm is scaled like a compact industrial manipulator, so the demo
camera and target field are tuned to a smaller meter-scale workspace instead of
the older oversized teaching-arm geometry.

### Runtime Loop

```text
x_d -> x -> v_des -> q_dot_cmd -> q_ref -> controller -> tau -> dynamics -> q
```

### What It Shows

- sequential target tracking in 3D on the 7-joint arm
- 15 randomly scattered targets per run
- active target highlighting
- end-effector trace through space
- default metric plots for:
  - Cartesian position error norm
  - desired end-effector speed norm
  - all-joint speed norm
  - minimum singular value
- clicking a joint chip switches the same four cards to that joint's angle, velocity, torque, and power
- each metric plot uses a rounded 5-tick y-axis ladder that resizes automatically
- the plots show grid lines only at the labeled major tick values
- the time axis uses 2-second ticks
- metric histories keep the last 15 seconds
- metric axes follow the visible range of the active view
- the active target advances once the tool-point error falls below the reach tolerance or the fallback step limit is hit
- the active target is highlighted by a blinking red marker while the blue target cloud omits that point until selection advances
- joint-specific charts appear immediately from buffered history instead of restarting from empty
- clicking the active joint chip again restores the original Cartesian graphs without clearing history
- the native window close button ends the demo cleanly, with `Esc` and `q` as fallbacks

## Joint-Space Step Response

This demo is a validation path for controller and dynamics behavior in isolation.
It uses the same dashboard layout, with the simulation still dominant on the
right and the metric cards stacked on the left.

### Runtime Loop

```text
q_ref -> controller -> tau -> dynamics -> integration -> q
```

### What It Shows

- controller response to a step input on the 7-joint chain
- default metric plots for:
  - joint 1 position error norm
  - all-joint velocity error norm
  - all-joint actuator torque norm
  - all-joint maximum joint error magnitude
- clicking a joint chip switches the same four cards to that joint's angle, velocity, torque, and power
- each metric plot uses a rounded 5-tick y-axis ladder that resizes automatically
- the plots show grid lines only at the labeled major tick values
- the time axis uses 2-second ticks
- metric histories keep the last 15 seconds
- metric axes follow the visible range of the active view
- clicking the active joint chip again returns to the original joint-space graphs without clearing history
- the same keyboard fallback applies here

## Demo Expectations

- The Cartesian demo should be the default showcase for the project.
- The joint-space demo should remain available for controller tuning and response inspection.
- Visualization should remain tied to the FK-derived geometry so the rendered arm matches the model.
- The demo window should launch as a standard resizable desktop window in a session with a usable display backend.
- The demos should not rely on a custom in-window exit button; the OS title-bar controls are the primary close path.
