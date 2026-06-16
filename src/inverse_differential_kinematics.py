import numpy as np
from typing import Sequence

try:
    # When used as part of the package
    from .forward_kinematics import Arm7DOFDH
    from .geometric_jacobian import position_jacobian
except ImportError:
    # When run directly from the src directory
    from forward_kinematics import Arm7DOFDH
    from geometric_jacobian import position_jacobian


def damped_pseudo_inverse(J: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """
    Compute the damped least-squares pseudo-inverse of a Jacobian.

    Parameters
    ----------
    J : ndarray, shape (m, n)
        Jacobian matrix.
    lam : float
        Damping factor (lambda). Larger values give more robustness near
        singularities but less accuracy.

    Returns
    -------
    J_pinv : ndarray, shape (n, m)
        Damped pseudo-inverse of J.
    """
    m, n = J.shape
    if lam <= 0.0:
        raise ValueError("lam must be positive.")
    if m >= n:
        lhs = J.T @ J + (lam**2) * np.eye(n)
        return np.linalg.solve(lhs, J.T)
    lhs = J @ J.T + (lam**2) * np.eye(m)
    return J.T @ np.linalg.solve(lhs, np.eye(m))


def saturate_norm(vector: Sequence[float], max_norm: float) -> np.ndarray:
    """
    Scale a vector so its Euclidean norm does not exceed ``max_norm``.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")

    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= max_norm or norm == 0.0:
        return v
    return v * (max_norm / norm)


def adaptive_damping(
    sigma_min_value: float,
    lambda_0: float = 1e-2,
    k_lambda: float = 3e-3,
    epsilon: float = 1e-3,
) -> float:
    """
    Compute the singularity-aware damping term used by the resolved-rate loop.
    """
    if lambda_0 <= 0.0:
        raise ValueError("lambda_0 must be positive.")
    if k_lambda < 0.0:
        raise ValueError("k_lambda must be non-negative.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    return float(lambda_0 + k_lambda / (sigma_min_value + epsilon))


def inverse_differential_kinematics(
    arm: Arm7DOFDH,
    q: Sequence[float],
    v_desired: np.ndarray,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    Compute joint velocities from a desired end-effector linear velocity.

    Parameters
    ----------
    arm : Arm7DOFDH
        Arm model.
    q : array_like, shape (7,)
        Current joint configuration.
    v_desired : ndarray, shape (3,)
        Desired tool-origin linear velocity in the base frame.
    lam : float
        Damping factor for the pseudo-inverse.

    Returns
    -------
    qdot : ndarray, shape (7,)
        Joint velocities realizing the desired spatial velocity in the
        least-squares sense.
    """
    v_desired = np.asarray(v_desired, dtype=float)
    if v_desired.shape != (3,):
        raise ValueError(f"Expected a 3-vector for v_desired, got shape {v_desired.shape}.")

    J_v = position_jacobian(arm, q)
    J_pinv = damped_pseudo_inverse(J_v, lam=lam)
    return J_pinv @ v_desired


def main() -> None:
    """
    Small demo: print a joint-velocity command for a sample spatial command.
    """
    arm = Arm7DOFDH()
    q = np.deg2rad([30.0, 20.0, -15.0, 40.0, -25.0, 15.0, 10.0])
    v_desired = np.array([0.1, 0.0, 0.0], dtype=float)
    qdot = inverse_differential_kinematics(arm, q, v_desired)
    np.set_printoptions(precision=3, suppress=True)
    print("qdot:")
    print(qdot)


if __name__ == "__main__":
    main()
