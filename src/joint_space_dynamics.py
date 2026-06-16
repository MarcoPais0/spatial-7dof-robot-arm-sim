import numpy as np
from typing import Sequence, Tuple

try:
    # When used as part of the package
    from .forward_kinematics import Arm7DOFDH
except ImportError:
    # When run directly from the src directory
    from forward_kinematics import Arm7DOFDH


class SimpleDynamics7DOF:
    """
    Minimal joint-space dynamics for the 7DOF arm.

    Assumptions:
    - Diagonal inertia and damping matrices.
    - Gravity torque uses a lightweight projected-link model that is the
      intended final v1 behavior for this project.
    """

    def __init__(
        self,
        arm: Arm7DOFDH,
        joint_inertias: Sequence[float] | None = None,
        joint_dampings: Sequence[float] | None = None,
        gravity: float = 9.81,
        link_masses: Sequence[float] | None = None,
    ) -> None:
        self.arm = arm
        n = arm.dof

        if joint_inertias is None:
            joint_inertias = [1.0] * n
        if joint_dampings is None:
            joint_dampings = [0.2] * n
        if link_masses is None:
            link_masses = [1.0] * n

        if len(joint_inertias) != n or len(joint_dampings) != n or len(link_masses) != n:
            raise ValueError("All parameter sequences must have length equal to DOF.")

        self.inertias = np.asarray(joint_inertias, dtype=float)
        self.dampings = np.asarray(joint_dampings, dtype=float)
        self.masses = np.asarray(link_masses, dtype=float)
        self.g_const = float(gravity)

    def mass_matrix(self) -> np.ndarray:
        """
        Return the diagonal mass matrix M(q) ≈ diag(inertias).
        """
        return np.diag(self.inertias)

    def damping_matrix(self) -> np.ndarray:
        """
        Return the diagonal damping matrix D ≈ diag(dampings).
        """
        return np.diag(self.dampings)

    def gravity_torque(self, q: Sequence[float]) -> np.ndarray:
        """
        Lightweight gravity torque approximation based on projected segment
        lengths.

        This is the final gravity model for the current version of the
        project. It intentionally trades physical fidelity for a compact and
        stable educational implementation.
        """
        n = self.arm.dof
        if len(q) != n:
            raise ValueError(f"Expected {n} joint angles, got {len(q)}.")

        q = np.asarray(q, dtype=float)
        g = self.g_const

        torques = np.zeros(n, dtype=float)
        lengths = np.asarray(getattr(self.arm, "segment_lengths", np.ones(n)), dtype=float)
        if lengths.shape != (n,):
            lengths = np.resize(lengths, n)

        # For each link, approximate a projected CoM contribution.
        for i in range(n):
            r_com = 0.5 * lengths[i]
            theta_com = float(np.sum(q[: i + 1]))
            contribution = self.masses[i] * g * r_com * np.cos(theta_com)
            torques[: i + 1] += contribution

        return torques

    def acceleration(
        self,
        q: Sequence[float],
        q_dot: Sequence[float],
        tau: Sequence[float],
    ) -> np.ndarray:
        """
        Compute joint accelerations from current state and torques.
        """
        q_dot = np.asarray(q_dot, dtype=float)
        tau = np.asarray(tau, dtype=float)

        M = self.mass_matrix()
        D = self.damping_matrix()
        g_tau = self.gravity_torque(q)

        # M q_ddot = tau - D q_dot - g(q)
        rhs = tau - D @ q_dot - g_tau
        return np.linalg.solve(M, rhs)

    def step(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        tau: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the dynamics forward using simple semi-implicit Euler.
        """
        q_ddot = self.acceleration(q, q_dot, tau)
        q_dot_new = q_dot + dt * q_ddot
        q_new = q + dt * q_dot_new
        return q_new, q_dot_new


def main() -> None:
    """
    Small smoke test for the dynamics class.
    """
    arm = Arm7DOFDH()
    dyn = SimpleDynamics7DOF(arm)

    q = np.zeros(7)
    q_dot = np.zeros(7)
    tau = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dt = 0.01

    for _ in range(10):
        q, q_dot = dyn.step(q, q_dot, tau, dt)
    print("q after 10 steps:", q)


# Backward compatibility alias for older imports.
SimpleDynamics4DOF = SimpleDynamics7DOF


if __name__ == "__main__":
    main()
