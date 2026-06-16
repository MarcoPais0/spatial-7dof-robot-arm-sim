import numpy as np
from dataclasses import dataclass
from typing import Sequence

try:
    # When used as part of the package
    from .forward_kinematics import Arm7DOFDH
except ImportError:
    # When run directly from the src directory
    from forward_kinematics import Arm7DOFDH


@dataclass(frozen=True)
class JacobianAnalysis:
    """
    SVD-based analysis of the position Jacobian J_v.

    Attributes
    ----------
    singular_values : ndarray, shape (3,)
        Singular values of J_v in descending order.
    rank : int
        Numerical rank computed from the singular values.
    min_singular_value : float
        Smallest singular value used by singularity-aware damping logic.
    condition_number : float
        Ratio between largest and smallest singular values, or inf if singular.
    manipulability : float
        Product of singular values, equivalent to sqrt(det(J_v J_v^T)).
    """

    singular_values: np.ndarray
    rank: int
    min_singular_value: float
    condition_number: float
    manipulability: float


def geometric_jacobian(arm: Arm7DOFDH, q: Sequence[float]) -> np.ndarray:
    """
    Compute the 6xn base-frame geometric Jacobian for the DH arm.

    The Jacobian is constructed numerically from the DH chain using the
    classic z-axis / cross-product method:
        v_i   = z_i-1 x (p_e - p_i-1)
        w_i   = z_i-1

    The upper block J_v maps joint velocities to tool-origin linear velocity.
    With the default tool frame aligned to the final wrist axis, the last
    column of J_v is zero: the terminal roll joint contributes orientation,
    not tool translation.

    Parameters
    ----------
    arm : Arm7DOFDH
        Arm model providing DH parameters.
    q : array_like, shape (4,)
        Joint angles.

    Returns
    -------
    J : (6, 4) ndarray
        Geometric Jacobian mapping joint velocities to end-effector spatial
        velocity [v; w].
    """
    if len(q) != arm.dof:
        raise ValueError(f"Expected {arm.dof} joint angles, got {len(q)}.")

    state = arm.frame_state(q)
    prev_frame_transforms = (np.eye(4),) + state.joint_transforms[:-1]
    p_e = state.tool_point
    J = np.zeros((6, arm.dof), dtype=float)

    for i, T_prev in enumerate(prev_frame_transforms):
        z_i = T_prev[:3, 2]
        p_i = T_prev[:3, 3]
        v_i = np.cross(z_i, p_e - p_i)
        w_i = z_i
        J[0:3, i] = v_i
        J[3:6, i] = w_i

    return J


def position_jacobian(arm: Arm7DOFDH, q: Sequence[float]) -> np.ndarray:
    """
    Return the 3x4 position Jacobian J_v(q) in the base frame.

    J_v is the primary Topic 4 block used by position-control and inverse
    differential-kinematics logic.
    """
    return geometric_jacobian(arm, q)[0:3, :]


def analyze_position_jacobian(
    arm: Arm7DOFDH,
    q: Sequence[float],
    tol: float = 1e-9,
) -> JacobianAnalysis:
    """
    Compute SVD-derived rank, conditioning, and manipulability for J_v(q).
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    J_v = position_jacobian(arm, q)
    singular_values = np.linalg.svd(J_v, compute_uv=False)
    rank = int(np.sum(singular_values > tol))
    min_singular_value = float(singular_values[-1])
    if min_singular_value > tol:
        condition_number = float(singular_values[0] / min_singular_value)
    else:
        condition_number = float("inf")
    manipulability = float(np.prod(singular_values))

    return JacobianAnalysis(
        singular_values=singular_values,
        rank=rank,
        min_singular_value=min_singular_value,
        condition_number=condition_number,
        manipulability=manipulability,
    )


def main() -> None:
    """
    Small demo: print J(q), J_v(q), and SVD analysis at a sample configuration.
    """
    arm = Arm7DOFDH()
    q = np.deg2rad([30.0, 20.0, -15.0, 40.0, -25.0, 15.0, 10.0])
    J = geometric_jacobian(arm, q)
    J_v = position_jacobian(arm, q)
    analysis = analyze_position_jacobian(arm, q)

    np.set_printoptions(precision=3, suppress=True)
    print("J(q):")
    print(J)
    print("J_v(q):")
    print(J_v)
    print("J_v singular values:", analysis.singular_values)
    print("J_v rank:", analysis.rank)
    print("J_v condition number:", analysis.condition_number)
    print("J_v manipulability:", analysis.manipulability)


if __name__ == "__main__":
    main()
