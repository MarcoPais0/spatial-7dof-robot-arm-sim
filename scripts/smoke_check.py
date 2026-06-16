#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forward_kinematics import Arm7DOFDH
from src.geometric_jacobian import analyze_position_jacobian, geometric_jacobian, position_jacobian
from src.inverse_differential_kinematics import adaptive_damping, inverse_differential_kinematics, saturate_norm
from src.joint_space_controller import JointSpacePDController, PDGains, simulate_joint_step_response
from src.joint_space_dynamics import SimpleDynamics7DOF


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    arm = Arm7DOFDH()
    q = np.deg2rad([30.0, 20.0, -15.0, 10.0, -12.0, 8.0, 5.0])

    state = arm.frame_state(q)
    _ensure(state.joint_7_transform.shape == (4, 4), "FK joint-7 transform has wrong shape.")
    _ensure(state.tool_transform.shape == (4, 4), "FK tool transform has wrong shape.")
    _ensure(len(state.joint_transforms) == 7, "FK joint transform count is wrong.")
    _ensure(state.joint_points.shape == (8, 3), "FK joint points have wrong shape.")
    _ensure(state.tool_point.shape == (3,), "FK tool point has wrong shape.")
    _ensure(np.all(np.isfinite(state.tool_point)), "FK tool point contains non-finite values.")

    q_roll = q.copy()
    q_roll[6] += 0.5
    rolled_state = arm.frame_state(q_roll)
    _ensure(
        np.allclose(state.tool_point, rolled_state.tool_point, atol=1e-9, rtol=1e-9),
        "Tool point should stay fixed when only q7 changes.",
    )

    J = geometric_jacobian(arm, q)
    J_v = position_jacobian(arm, q)
    analysis = analyze_position_jacobian(arm, q)
    _ensure(J.shape == (6, 7), "Geometric Jacobian has wrong shape.")
    _ensure(J_v.shape == (3, 7), "Position Jacobian has wrong shape.")
    _ensure(analysis.singular_values.shape == (3,), "Jacobian analysis has wrong singular-value shape.")
    _ensure(np.all(np.isfinite(J)), "Geometric Jacobian contains non-finite values.")
    _ensure(np.all(np.isfinite(J_v)), "Position Jacobian contains non-finite values.")
    _ensure(np.allclose(J_v[:, 6], 0.0, atol=1e-9, rtol=1e-9), "Default geometry should make the q7 position column zero.")

    v_raw = np.array([0.8, 0.8, 0.0], dtype=float)
    v_desired = saturate_norm(v_raw, 0.5)
    _ensure(np.isclose(np.linalg.norm(v_desired), 0.5, atol=1e-9, rtol=1e-9), "Cartesian speed saturation should limit the vector norm.")

    lam_near = adaptive_damping(0.01)
    lam_far = adaptive_damping(1.0)
    _ensure(lam_near > lam_far, "Adaptive damping should increase as the smallest singular value decreases.")

    q_dot_cmd = inverse_differential_kinematics(arm, q, v_desired, lam=1e-2)
    _ensure(q_dot_cmd.shape == (7,), "Inverse kinematics returned the wrong shape.")
    _ensure(np.all(np.isfinite(q_dot_cmd)), "Inverse kinematics returned non-finite values.")

    dyn = SimpleDynamics7DOF(arm)
    controller = JointSpacePDController(
        gains=[
            PDGains(kp=5.0, kd=1.0),
            PDGains(kp=4.5, kd=0.9),
            PDGains(kp=4.0, kd=0.8),
            PDGains(kp=3.5, kd=0.7),
            PDGains(kp=3.0, kd=0.6),
            PDGains(kp=2.5, kd=0.5),
            PDGains(kp=2.0, kd=0.4),
        ],
        dyn=dyn,
    )

    tau_gravity = controller.compute_torque(
        q=q,
        q_dot=np.zeros(7),
        q_ref=q,
        q_dot_ref=np.zeros(7),
    )
    _ensure(
        np.allclose(tau_gravity, dyn.gravity_torque(q), atol=1e-9, rtol=1e-9),
        "Controller should add gravity compensation directly.",
    )

    tau = controller.compute_torque(
        q=q,
        q_dot=np.zeros(7),
        q_ref=q + np.deg2rad([20.0, -10.0, 15.0, 5.0, -8.0, 6.0, 4.0]),
        q_dot_ref=np.zeros(7),
    )
    _ensure(tau.shape == (7,), "Controller returned the wrong torque shape.")
    _ensure(np.all(np.isfinite(tau)), "Controller returned non-finite torques.")
    _ensure(
        np.max(np.abs(tau)) <= controller.tau_limit + 1e-9,
        "Controller torque saturation is not enforcing the limit.",
    )

    q_next, q_dot_next = dyn.step(q, np.zeros(7), tau, dt=0.01)
    _ensure(q_next.shape == (7,), "Dynamics returned the wrong q shape.")
    _ensure(q_dot_next.shape == (7,), "Dynamics returned the wrong q_dot shape.")
    _ensure(np.all(np.isfinite(q_next)), "Dynamics returned non-finite q values.")
    _ensure(np.all(np.isfinite(q_dot_next)), "Dynamics returned non-finite q_dot values.")

    ts, qs, qdots = simulate_joint_step_response(
        dyn=dyn,
        controller=controller,
        q_init=np.zeros(7),
        qdot_init=np.zeros(7),
        q_ref=np.deg2rad([10.0, -5.0, 15.0, 0.0, -8.0, 6.0, 4.0]),
        dt=0.01,
        t_final=0.05,
    )
    _ensure(ts.ndim == 1 and qs.shape[1] == 7 and qdots.shape[1] == 7, "Step-response simulation returned inconsistent shapes.")
    _ensure(np.all(np.isfinite(qs)), "Step-response simulation produced non-finite joint positions.")
    _ensure(np.all(np.isfinite(qdots)), "Step-response simulation produced non-finite joint velocities.")

    print("Smoke check passed.")


if __name__ == "__main__":
    main()
