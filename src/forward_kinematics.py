from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DHLink:
    """
    Standard Denavit-Hartenberg (DH) parameters for a single revolute joint.

    All joints are assumed revolute. The parameters (a, alpha, d, theta_offset)
    are defined using the standard DH convention for T_(i-1)^i. The actual
    joint angle is theta_i = q_i + theta_offset_i.
    """

    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0


@dataclass(frozen=True)
class ArmFrameState:
    """
    Frame-wise forward-kinematics result for the 7R chain.

    Attributes
    ----------
    joint_transforms : tuple of ndarray
        Ordered base-to-joint transforms for frames {1} through {7}.
    tool_transform : (4, 4) ndarray
        Homogeneous transform from the base frame to the tool frame.
    joint_points : (8, 3) ndarray
        Frame origins ordered as [base, 1, 2, 3, 4, 5, 6, 7].
    tool_point : (3,) ndarray
        Origin of the tool frame expressed in the base frame.
    """

    joint_transforms: tuple[np.ndarray, ...]
    tool_transform: np.ndarray
    joint_points: np.ndarray
    tool_point: np.ndarray

    @property
    def joint_7_transform(self) -> np.ndarray:
        """
        Homogeneous transform T_0^7 for the final joint frame.
        """
        return self.joint_transforms[-1]


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute the standard DH homogeneous transform T_(i-1)^i.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class Arm7DOFDH:
    """
    Spatial 7DOF revolute manipulator defined by a compact industrial-arm DH model.

    This is the authoritative geometry definition for:
    - frames {0} through {7}, and {tool}
    - seven revolute joints (R-R-R-R-R-R-R)
    - the standard DH links T_(i-1)^i
    - the explicit tool transform T_7^tool

    The arm configuration is:
        q = [q1, q2, q3, q4, q5, q6, q7]^T

    Default DH parameters:
        a_i     = [0, 0, 0, 0, 0, 0, 0]
        alpha_i = [-pi/2, pi/2, pi/2, -pi/2, -pi/2, pi/2, 0]
        d_i     = [0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.126]
        theta_i = q_i + theta_offset_i

    The default tool transform keeps the tool frame aligned with z7 and
    coincident with the final joint frame unless an additional tool offset is
    provided explicitly.
    """

    BASE_FRAME = "0"
    JOINT_FRAMES = ("1", "2", "3", "4", "5", "6", "7")
    TOOL_FRAME = "tool"

    DEFAULT_A = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    DEFAULT_ALPHA = (-np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, 0.0)
    DEFAULT_D = (0.34, 0.0, 0.40, 0.0, 0.40, 0.0, 0.126)
    DEFAULT_SEGMENT_LENGTHS = (0.34, 0.40, 0.40, 0.126, 0.126, 0.10, 0.08)

    def __init__(
        self,
        theta_offsets: Sequence[float] | None = None,
        T_7_tool: np.ndarray | None = None,
    ) -> None:
        if theta_offsets is None:
            theta_offsets = (0.0,) * 7
        if len(theta_offsets) != 7:
            raise ValueError("theta_offsets must contain exactly seven values.")

        self.theta_offsets = tuple(float(offset) for offset in theta_offsets)
        self.links = [
            DHLink(
                a=a_i,
                alpha=alpha_i,
                d=d_i,
                theta_offset=theta_offset,
            )
            for a_i, alpha_i, d_i, theta_offset in zip(
                self.DEFAULT_A,
                self.DEFAULT_ALPHA,
                self.DEFAULT_D,
                self.theta_offsets,
                strict=True,
            )
        ]
        self.segment_lengths = np.asarray(self.DEFAULT_SEGMENT_LENGTHS, dtype=float)

        if T_7_tool is None:
            T_7_tool = np.eye(4, dtype=float)

        self.T_7_tool = np.asarray(T_7_tool, dtype=float).copy()
        if self.T_7_tool.shape != (4, 4):
            raise ValueError("T_7_tool must have shape (4, 4).")
        if not np.all(np.isfinite(self.T_7_tool)):
            raise ValueError("T_7_tool must contain only finite values.")
        if not np.allclose(self.T_7_tool[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("T_7_tool must be a homogeneous transform.")
        R_7_tool = self.T_7_tool[:3, :3]
        if not np.allclose(R_7_tool.T @ R_7_tool, np.eye(3)):
            raise ValueError("T_7_tool rotation must be orthonormal.")
        if not np.isclose(np.linalg.det(R_7_tool), 1.0):
            raise ValueError("T_7_tool rotation must be right-handed.")
        if not np.allclose(R_7_tool[:, 2], np.array([0.0, 0.0, 1.0])):
            raise ValueError("T_7_tool must keep the tool z-axis aligned with z7.")
        tool_offset = self.T_7_tool[:3, 3]
        if not np.allclose(tool_offset[:2], np.zeros(2)):
            raise ValueError("T_7_tool translation must lie on the z7 axis.")
        if tool_offset[2] < 0.0:
            raise ValueError("T_7_tool translation must be non-negative along z7.")

    @property
    def dof(self) -> int:
        return len(self.links)

    @property
    def reach(self) -> float:
        """
        Compact scalar reach estimate used by the demo for workspace scaling.
        """
        return float(np.sum(self.DEFAULT_D[::2]))

    def frame_state(self, q: Sequence[float]) -> ArmFrameState:
        """
        Evaluate the base, joint, and tool frames for a joint configuration.
        """
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != (self.dof,):
            raise ValueError(f"Expected {self.dof} joint angles, got shape {q_arr.shape}.")

        T = np.eye(4, dtype=float)
        joint_transforms = []
        points = [T[:3, 3].copy()]
        for qi, link in zip(q_arr, self.links, strict=True):
            theta = qi + link.theta_offset
            T = T @ dh_transform(link.a, link.alpha, link.d, theta)
            joint_transforms.append(T.copy())
            points.append(T[:3, 3].copy())

        tool_transform = joint_transforms[-1] @ self.T_7_tool
        joint_points = np.stack(points, axis=0)
        tool_point = tool_transform[:3, 3].copy()

        return ArmFrameState(
            joint_transforms=tuple(joint_transforms),
            tool_transform=tool_transform,
            joint_points=joint_points,
            tool_point=tool_point,
        )


# Backward compatibility alias for older imports.
Arm4DOFDH = Arm7DOFDH


def main() -> None:
    """
    Small sanity check: print the final joint and tool transforms for a sample q.
    """
    arm = Arm7DOFDH()
    q = np.deg2rad([30.0, 20.0, -15.0, 40.0, -25.0, 15.0, 10.0])
    state = arm.frame_state(q)
    np.set_printoptions(precision=3, suppress=True)
    print("T_0_7(q):")
    print(state.joint_7_transform)
    print("T_0_tool(q):")
    print(state.tool_transform)


if __name__ == "__main__":
    main()
