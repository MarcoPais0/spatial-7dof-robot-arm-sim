# AGENTS.md

## Purpose

This repository is organized around a small simulation codebase and a separate documentation set. Keep the top-level README brief and put technical detail in `docs/`.

## Repository Layout

- `src/` contains the simulation modules
- `scripts/` contains lightweight validation and utility entrypoints
- `README.md` contains only essential entry information
- `AGENTS.md` contains repository working instructions
- `docs/` contains the technical knowledge base and implementation notes

## Working Rules

- Keep documentation aligned with the current code, especially the DH model, Jacobian, dynamics, controller, and demos.
- Update the relevant topic document when changing behavior or assumptions.
- All business, design, and architecture rules must be documented in the proper location, usually the relevant `docs/` topic file or a new topic file when needed.
- When a change affects runtime behavior, windowing, or user workflow, perform a doc-code coherence pass on the affected docs before treating the work as complete.
- Do not expand the README into a long narrative; link into `docs/` instead.
- Prefer one topic per doc so the material can be reused later in a formal report.
- Preserve terminology consistently: frames, DH geometry, tool transform, `J_v`, simplified dynamics, and simulation loop.

## Reading Order

1. `docs/README.md`
2. `docs/overview.md`
3. `docs/kinematics.md`
4. `docs/jacobian-ik.md`
5. `docs/dynamics-control.md`
6. `docs/demos.md`
7. `docs/assumptions-limitations.md`

## Implementation Expectations

- The public technical story should match the code: `forward_kinematics`, `geometric_jacobian`, `inverse_differential_kinematics`, `joint_space_dynamics`, `joint_space_controller`, and `robot_arm_3d_demo`.
- If a change alters a mathematical assumption or a demo behavior, document it in `docs/` before treating the work as complete.
- If new topics appear, add a new markdown file in `docs/` rather than expanding an unrelated one.
- `scripts/smoke_check.py` is the expected repository validation command for checking that the app and core math pipeline still run.
