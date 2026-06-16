# Overview

## Purpose

The project models a spatial 7DOF robotic arm and provides a compact simulation stack for kinematic analysis, resolved-rate motion, simplified dynamics, joint-space control, and 3D visualization.

## Scope

- The arm uses standard DH geometry with seven revolute joints.
- The task-space focus is end-effector position control in 3D.
- End-effector roll exists as the redundant motion about the final wrist axis but is not the primary controlled output.
- The implementation favors a clear educational model over a full rigid-body dynamics engine.

## Core Interfaces

- `FK`: forward kinematics and frame/point extraction
- `Jacobian`: geometric Jacobian and singularity analysis
- `IK`: inverse differential kinematics for tool-position motion
- `Dynamics`: simplified joint-space dynamics
- `Controller`: joint-space PD tracking with gravity compensation and torque saturation

## System Map

```text
DH geometry
  -> forward kinematics
  -> geometric Jacobian
  -> inverse differential kinematics
  -> controller
  -> simplified dynamics
  -> numerical integration
  -> visualization and demos
```

## Units

- Joint angles in radians
- Angular velocities in rad/s
- Lengths in meters
- Linear velocities in m/s
- Torques in N m
