from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque

import numpy as np

try:
    from .forward_kinematics import Arm7DOFDH
    from .geometric_jacobian import analyze_position_jacobian
    from .inverse_differential_kinematics import adaptive_damping, inverse_differential_kinematics, saturate_norm
    from .joint_space_controller import JointSpacePDController, PDGains
    from .joint_space_dynamics import SimpleDynamics7DOF
except ImportError:
    from forward_kinematics import Arm7DOFDH
    from geometric_jacobian import analyze_position_jacobian
    from inverse_differential_kinematics import adaptive_damping, inverse_differential_kinematics, saturate_norm
    from joint_space_controller import JointSpacePDController, PDGains
    from joint_space_dynamics import SimpleDynamics7DOF

try:
    from .gui_environment import collect_runtime_issues, configure_qt_plugin_paths, format_runtime_issues
except ImportError:
    from gui_environment import collect_runtime_issues, configure_qt_plugin_paths, format_runtime_issues


class DemoStartupError(RuntimeError):
    pass


try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import OpenGL.GL as _gl_module
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    _QT_AVAILABLE = False
    QtCore = QtGui = QtWidgets = pg = gl = None  # type: ignore[assignment]
else:
    _QT_AVAILABLE = True


WINDOW_BACKGROUND = "#1c1c1e"
CARD_BACKGROUND = "#232327"
HEADER_BACKGROUND = "#2a2a2e"
PLOT_BACKGROUND = "#17171a"
TEXT_COLOR = "#f5f5f7"
MUTED_TEXT_COLOR = "#d1d1d6"
GRID_COLOR = "#3a3a3c"
ACCENT_RED = "#ff453a"
ACCENT_BLUE = "#0a84ff"
ACCENT_GREEN = "#30d158"
ACCENT_ORANGE = "#ff9f0a"
ACCENT_CYAN = "#64d2ff"
PLANAR_SURFACE = "#3a3a3c"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    description: str
    summary: str
    color: str


@dataclass
class DemoSnapshot:
    time: float
    arm_points: np.ndarray
    trace_points: np.ndarray
    inactive_targets: np.ndarray | None
    active_target: np.ndarray | None
    active_target_visible: bool
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    joint_powers: np.ndarray
    metrics: dict[str, float]


def _default_pd_gains(dof: int) -> list[PDGains]:
    kp_values = np.linspace(5.5, 2.0, dof)
    kd_values = np.linspace(1.1, 0.4, dof)
    return [PDGains(kp=float(kp), kd=float(kd)) for kp, kd in zip(kp_values, kd_values, strict=True)]


class BaseDemoSession:
    mode_title = "7DOF robotic arm"
    metric_specs: tuple[MetricSpec, ...] = ()
    joint_metric_specs: tuple[MetricSpec, ...] = (
        MetricSpec(
            "angle",
            "Angle",
            "Angular position of the selected joint in radians.",
            "Where the selected joint is rotated to.",
            ACCENT_RED,
        ),
        MetricSpec(
            "velocity",
            "Velocity",
            "Angular velocity of the selected joint in radians per second.",
            "How fast the selected joint is moving.",
            ACCENT_BLUE,
        ),
        MetricSpec(
            "torque",
            "Torque",
            "Commanded torque applied to the selected joint in newton-meters.",
            "How hard the selected joint is being driven.",
            ACCENT_GREEN,
        ),
        MetricSpec(
            "power",
            "Power",
            "Instantaneous mechanical power of the selected joint, computed as torque times angular velocity.",
            "How much work the selected joint is doing right now.",
            ACCENT_ORANGE,
        ),
    )
    dt = 0.02

    def __init__(self) -> None:
        self.arm = Arm7DOFDH()
        self.dyn = SimpleDynamics7DOF(self.arm)
        self.controller = JointSpacePDController(
            gains=_default_pd_gains(self.arm.dof),
            dyn=self.dyn,
            tau_limit=50.0,
        )
        self.time = 0.0
        self.current_tau = np.zeros(self.arm.dof, dtype=float)
        self.current_power = np.zeros(self.arm.dof, dtype=float)

    def step(self, dt: float) -> DemoSnapshot:
        raise NotImplementedError

    def snapshot(self) -> DemoSnapshot:
        raise NotImplementedError


class CartesianTrackingSession(BaseDemoSession):
    mode_title = "Cartesian tracking"
    metric_specs = (
        MetricSpec(
            "position_error_norm",
            "Cartesian position error norm",
            "Euclidean norm of the Cartesian position error between the active target and the tool point.",
            "How far the tool is from the target.",
            ACCENT_RED,
        ),
        MetricSpec(
            "commanded_speed_norm",
            "Desired end-effector speed norm",
            "Magnitude of the desired end-effector velocity vector sent into inverse kinematics.",
            "How fast the end effector is being asked to move.",
            ACCENT_BLUE,
        ),
        MetricSpec(
            "joint_speed_norm",
            "All-joint speed norm",
            "Euclidean norm of the joint velocity vector across all seven joints.",
            "How much the arm's joints are moving overall.",
            ACCENT_GREEN,
        ),
        MetricSpec(
            "min_singular_value",
            "Minimum singular value",
            "Smallest singular value of the position Jacobian at the current configuration.",
            "How close the arm is to a singular pose.",
            ACCENT_ORANGE,
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.cartesian_position_gain = 1.4175
        self.cartesian_velocity_limit = 0.9
        self.cartesian_lambda_0 = 1e-2
        self.cartesian_lambda_gain = 3e-3
        self.cartesian_lambda_epsilon = 1e-3
        self.cartesian_target_tolerance = 0.01
        self.cartesian_target_max_steps = 490
        self.cartesian_target_count = 15
        self.cartesian_target_seed = 7
        self.q = np.zeros(self.arm.dof, dtype=float)
        self.q_dot = np.zeros(self.arm.dof, dtype=float)
        self.targets = self._generate_targets(self.cartesian_target_count)
        self.active_index = 0
        self.current_target = self.targets[self.active_index].copy()
        self.target_steps = 0
        self.blink_phase = 0.0
        self.blink_period = 0.6
        self.blink_on_duration = 0.3
        self.trace_points = np.repeat(self.arm.frame_state(self.q).tool_point[None, :], 180, axis=0)

    def _generate_targets(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(self.cartesian_target_seed)
        targets: list[np.ndarray] = []
        reach = self.arm.reach
        lower = np.array([0.10, -0.45, 0.15], dtype=float)
        upper = np.array([0.95, 0.45, 1.10], dtype=float)
        min_radius = 0.20
        max_radius = reach * 0.95

        while len(targets) < count:
            candidate = rng.uniform(lower, upper)
            radius = np.linalg.norm(candidate)
            if min_radius <= radius <= max_radius:
                targets.append(candidate)

        return np.asarray(targets, dtype=float)

    def _current_arm_state(self):
        return self.arm.frame_state(self.q)

    def _update_trace(self, point: np.ndarray) -> None:
        self.trace_points[:] = np.roll(self.trace_points, -1, axis=0)
        self.trace_points[-1] = point

    def _advance_target(self) -> None:
        self.active_index = (self.active_index + 1) % len(self.targets)
        self.current_target = self.targets[self.active_index].copy()
        self.target_steps = 0
        self.blink_phase = 0.0

    def snapshot(self) -> DemoSnapshot:
        state = self._current_arm_state()
        inactive_targets = np.delete(self.targets, self.active_index, axis=0) if len(self.targets) else None
        return DemoSnapshot(
            time=self.time,
            arm_points=state.joint_points,
            trace_points=self.trace_points,
            inactive_targets=inactive_targets,
            active_target=self.current_target,
            active_target_visible=self.blink_phase < self.blink_on_duration,
            joint_positions=self.q.copy(),
            joint_velocities=self.q_dot.copy(),
            joint_torques=self.current_tau.copy(),
            joint_powers=self.current_power.copy(),
            metrics={
                "position_error_norm": 0.0,
                "commanded_speed_norm": 0.0,
                "joint_speed_norm": 0.0,
                "min_singular_value": 0.0,
            },
        )

    def step(self, dt: float) -> DemoSnapshot:
        state = self._current_arm_state()
        tool_point = state.tool_point
        position_error = self.current_target - tool_point
        error_norm = float(np.linalg.norm(position_error))

        v_des_raw = self.cartesian_position_gain * position_error
        v_des = saturate_norm(v_des_raw, self.cartesian_velocity_limit)

        analysis = analyze_position_jacobian(self.arm, self.q)
        lam = adaptive_damping(
            analysis.min_singular_value,
            lambda_0=self.cartesian_lambda_0,
            k_lambda=self.cartesian_lambda_gain,
            epsilon=self.cartesian_lambda_epsilon,
        )
        q_dot_cmd = inverse_differential_kinematics(
            arm=self.arm,
            q=self.q,
            v_desired=v_des,
            lam=lam,
        )
        q_ref = self.q + dt * q_dot_cmd
        tau = self.controller.compute_torque(
            q=self.q,
            q_dot=self.q_dot,
            q_ref=q_ref,
            q_dot_ref=q_dot_cmd,
        )
        self.q, self.q_dot = self.dyn.step(self.q, self.q_dot, tau, dt)
        self.time += dt

        new_state = self.arm.frame_state(self.q)
        self._update_trace(new_state.tool_point)

        self.blink_phase = (self.blink_phase + dt) % self.blink_period
        self.target_steps += 1
        if error_norm <= self.cartesian_target_tolerance or self.target_steps >= self.cartesian_target_max_steps:
            self._advance_target()

        self.current_tau = tau.copy()
        self.current_power = self.current_tau * self.q_dot
        inactive_targets = np.delete(self.targets, self.active_index, axis=0) if len(self.targets) else None
        return DemoSnapshot(
            time=self.time,
            arm_points=new_state.joint_points,
            trace_points=self.trace_points,
            inactive_targets=inactive_targets,
            active_target=self.current_target,
            active_target_visible=self.blink_phase < self.blink_on_duration,
            joint_positions=self.q.copy(),
            joint_velocities=self.q_dot.copy(),
            joint_torques=self.current_tau.copy(),
            joint_powers=self.current_power.copy(),
            metrics={
                "position_error_norm": error_norm,
                "commanded_speed_norm": float(np.linalg.norm(v_des)),
                "joint_speed_norm": float(np.linalg.norm(self.q_dot)),
                "min_singular_value": float(analysis.min_singular_value),
            },
        )


class JointStepSession(BaseDemoSession):
    mode_title = "Joint-space validation"
    metric_specs = (
        MetricSpec(
            "position_error_norm",
            "Joint 1 position error norm",
            "Absolute position error of joint 1 relative to its commanded reference.",
            "How far joint 1 is from its target.",
            ACCENT_RED,
        ),
        MetricSpec(
            "torque_norm",
            "All-joint actuator torque norm",
            "Euclidean norm of the commanded joint torque vector across all seven joints.",
            "How much total torque the arm is using.",
            ACCENT_BLUE,
        ),
        MetricSpec(
            "velocity_error_norm",
            "All-joint velocity error norm",
            "Euclidean norm of the joint velocity error vector across all seven joints.",
            "How fast the joints are moving away from their target speeds.",
            ACCENT_GREEN,
        ),
        MetricSpec(
            "max_error",
            "All-joint maximum joint error magnitude",
            "Maximum absolute joint position error across all seven joints.",
            "The biggest joint error at this moment.",
            ACCENT_ORANGE,
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.q_ref = np.deg2rad([25.0, 20.0, -15.0, 10.0, -12.0, 8.0, 5.0])
        self.q = np.zeros(self.arm.dof, dtype=float)
        self.q_dot = np.zeros(self.arm.dof, dtype=float)
        self.trace_points = np.repeat(self.arm.frame_state(self.q).tool_point[None, :], 180, axis=0)
        self.current_tau = self.controller.compute_torque(
            q=self.q,
            q_dot=self.q_dot,
            q_ref=self.q_ref,
            q_dot_ref=np.zeros_like(self.q),
        )
        self.current_power = self.current_tau * self.q_dot

    def _current_arm_state(self):
        return self.arm.frame_state(self.q)

    def _update_trace(self, point: np.ndarray) -> None:
        self.trace_points[:] = np.roll(self.trace_points, -1, axis=0)
        self.trace_points[-1] = point

    def snapshot(self) -> DemoSnapshot:
        state = self._current_arm_state()
        return DemoSnapshot(
            time=self.time,
            arm_points=state.joint_points,
            trace_points=self.trace_points,
            inactive_targets=None,
            active_target=None,
            active_target_visible=False,
            joint_positions=self.q.copy(),
            joint_velocities=self.q_dot.copy(),
            joint_torques=self.current_tau.copy(),
            joint_powers=self.current_power.copy(),
            metrics={
                "position_error_norm": 0.0,
                "torque_norm": 0.0,
                "velocity_error_norm": 0.0,
                "max_error": 0.0,
            },
        )

    def step(self, dt: float) -> DemoSnapshot:
        tau = self.controller.compute_torque(
            q=self.q,
            q_dot=self.q_dot,
            q_ref=self.q_ref,
            q_dot_ref=np.zeros_like(self.q),
        )
        self.q, self.q_dot = self.dyn.step(self.q, self.q_dot, tau, dt)
        self.time += dt
        state = self.arm.frame_state(self.q)
        self._update_trace(state.tool_point)

        self.current_tau = tau.copy()
        self.current_power = self.current_tau * self.q_dot
        position_error = self.q_ref - self.q
        return DemoSnapshot(
            time=self.time,
            arm_points=state.joint_points,
            trace_points=self.trace_points,
            inactive_targets=None,
            active_target=None,
            active_target_visible=False,
            joint_positions=self.q.copy(),
            joint_velocities=self.q_dot.copy(),
            joint_torques=self.current_tau.copy(),
            joint_powers=self.current_power.copy(),
            metrics={
                "position_error_norm": float(abs(position_error[0])),
                "torque_norm": float(np.linalg.norm(tau)),
                "velocity_error_norm": float(np.linalg.norm(-self.q_dot)),
                "max_error": float(np.max(np.abs(position_error))),
            },
        )


if _QT_AVAILABLE:

    class ElidedMetricTitleLabel(QtWidgets.QLabel):
        def __init__(self, text: str) -> None:
            super().__init__(text)
            self._full_text = text

        def setFullText(self, text: str) -> None:
            self._full_text = text
            self._update_elision()

        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
            super().resizeEvent(event)
            self._update_elision()

        def _update_elision(self) -> None:
            metrics = QtGui.QFontMetrics(self.font())
            available = max(0, self.width())
            self.setText(metrics.elidedText(self._full_text, QtCore.Qt.TextElideMode.ElideRight, available))


    class TooltipTriggerLabel(QtWidgets.QLabel):
        def __init__(self, text: str = "") -> None:
            super().__init__(text)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)


    class JointSelectorChip(QtWidgets.QLabel):
        clicked = QtCore.Signal(int)

        def __init__(self, joint_index: int, text: str) -> None:
            super().__init__(text)
            self.joint_index = joint_index
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.clicked.emit(self.joint_index)
            super().mousePressEvent(event)


    class TimeAxisItem(pg.AxisItem):
        def __init__(self, *args, fixed_spacing: float | None = None, fixed_offset: float = 0.0, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fixed_spacing = fixed_spacing
            self.fixed_offset = fixed_offset

        def tickValues(self, minVal, maxVal, size):  # type: ignore[override]
            if self.orientation in {"bottom", "top"} and self.fixed_spacing is not None:
                spacing = float(self.fixed_spacing)
                if spacing <= 0.0:
                    return []
                start = np.floor((float(minVal) - self.fixed_offset) / spacing) * spacing + self.fixed_offset
                ticks = np.arange(start, float(maxVal) + spacing * 0.5, spacing, dtype=float)
                ticks = ticks[(ticks >= float(minVal) - 1e-9) & (ticks <= float(maxVal) + 1e-9)]
                return [(spacing, ticks.tolist())]

            values = super().tickValues(minVal, maxVal, size)
            if not values:
                return values
            major_spacing, major_values = max(values, key=lambda item: float(item[0]))
            return [(major_spacing, major_values)]

        def tickStrings(self, values: list[float], scale: float, spacing: float):  # type: ignore[override]
            return [f"{float(value) * scale:.0f}" for value in values]


    class AdaptiveValueAxisItem(pg.AxisItem):
        def __init__(self, *args, tick_count: int = 5, minimum_step: float = 0.1, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.tick_count = tick_count
            self.minimum_step = minimum_step
            self.current_step = 0.2
            self.current_lower = 0.0
            self.current_upper = self.current_step * (self.tick_count - 1)

        def update_band(self, visible_min: float, visible_max: float) -> tuple[float, float]:
            lo = float(min(visible_min, visible_max))
            hi = float(max(visible_min, visible_max))
            if np.isfinite(lo) and np.isfinite(hi):
                span = hi - lo
                step = max(self.minimum_step, math.ceil((span / max(self.tick_count - 1, 1)) / self.minimum_step) * self.minimum_step)
                while True:
                    lower = math.floor(lo / step) * step
                    upper = lower + step * (self.tick_count - 1)
                    if upper >= hi - 1e-9:
                        break
                    step = round(step + self.minimum_step, 10)
                self.current_step = step
                self.current_lower = lower
                self.current_upper = upper
                self.picture = None
                self.update()
            return self.current_lower, self.current_upper

        def tickValues(self, minVal, maxVal, size):  # type: ignore[override]
            step = float(self.current_step)
            lower = float(self.current_lower)
            ticks = [lower + step * i for i in range(self.tick_count)]
            return [(step, ticks)]

        def tickStrings(self, values: list[float], scale: float, spacing: float):  # type: ignore[override]
            return [f"{float(value) * scale:.1f}" for value in values]


    class MetricCard(QtWidgets.QFrame):
        def __init__(
            self,
            general_spec: MetricSpec,
            joint_spec: MetricSpec,
            general_value_getter: Callable[[DemoSnapshot], float],
            joint_value_getter: Callable[[DemoSnapshot], np.ndarray],
            joint_count: int,
            history_seconds: float = 15.0,
        ) -> None:
            super().__init__()
            self.setObjectName("MetricCard")
            self.general_spec = general_spec
            self.joint_spec = joint_spec
            self.general_value_getter = general_value_getter
            self.joint_value_getter = joint_value_getter
            self.joint_count = joint_count
            self.history_seconds = history_seconds
            self.time_tick_spacing = 2.0
            self.times: Deque[float] = deque()
            self.general_values: Deque[float] = deque()
            self.joint_values: list[Deque[float]] = [deque() for _ in range(self.joint_count)]
            self.selected_joint_index: int | None = None
            self._tooltip_text = f"{self.general_spec.description}\n{self.general_spec.summary}"
            self._tooltip_delay_ms = 500
            self._tooltip_anchor: QtWidgets.QWidget | None = None
            self._tooltip_timer = QtCore.QTimer(self)
            self._tooltip_timer.setSingleShot(True)
            self._tooltip_timer.timeout.connect(self._show_hover_tooltip)

            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)

            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(10)

            title_block = QtWidgets.QWidget()
            title_block_layout = QtWidgets.QHBoxLayout(title_block)
            title_block_layout.setContentsMargins(0, 0, 0, 0)
            title_block_layout.setSpacing(6)

            self.title_label = ElidedMetricTitleLabel(self.general_spec.title)
            self.title_label.setObjectName("MetricTitle")
            self.title_label.setWordWrap(False)
            self.title_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
            self.title_label.setMouseTracking(True)
            self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.title_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.title_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

            self.info_label = TooltipTriggerLabel("i")
            self.info_label.setObjectName("MetricInfo")
            self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.info_label.setFixedSize(18, 18)
            self.info_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.info_label.installEventFilter(self)
            self.title_label.installEventFilter(self)

            self.value_label = QtWidgets.QLabel("0.00")
            self.value_label.setObjectName("MetricValue")

            title_font = QtGui.QFont()
            title_font.setPointSize(14)
            title_font.setBold(False)
            value_font = QtGui.QFont()
            value_font.setPointSize(14)
            value_font.setBold(False)
            axis_font = QtGui.QFont()
            axis_font.setPointSize(11)
            axis_font.setBold(False)
            self.title_label.setFont(title_font)
            self.value_label.setFont(value_font)
            self.title_label.setFont(title_font)
            value_metrics = QtGui.QFontMetrics(value_font)
            self.value_label.setMinimumWidth(value_metrics.horizontalAdvance("-00000.00"))

            title_block_layout.addWidget(self.info_label, 0)
            title_block_layout.addWidget(self.title_label, 1)

            header_layout.addWidget(title_block, 1)
            header_layout.addWidget(self.value_label, 0, QtCore.Qt.AlignmentFlag.AlignRight)

            self.value_axis = AdaptiveValueAxisItem("left")
            plot_item = pg.PlotItem(
                axisItems={
                    "left": self.value_axis,
                    "bottom": TimeAxisItem("bottom", fixed_spacing=self.time_tick_spacing),
                }
            )
            self.plot = pg.PlotWidget(plotItem=plot_item)
            self.plot.setObjectName("MetricPlot")
            self.plot.setBackground(PLOT_BACKGROUND)
            self.plot.setMenuEnabled(False)
            self.plot.hideButtons()
            self.plot.showGrid(x=True, y=True, alpha=0.35)
            self.plot.setContentsMargins(0, 0, 0, 0)
            plot_item.showAxis("left")
            left_axis = plot_item.getAxis("left")
            left_axis.setPen(pg.mkPen(TEXT_COLOR))
            left_axis.setTextPen(pg.mkPen(MUTED_TEXT_COLOR))
            left_axis.setStyle(tickFont=axis_font)
            bottom_axis = plot_item.getAxis("bottom")
            bottom_axis.setPen(pg.mkPen(TEXT_COLOR))
            bottom_axis.setTextPen(pg.mkPen(MUTED_TEXT_COLOR))
            bottom_axis.setStyle(tickFont=axis_font)
            plot_item.setYRange(self.value_axis.current_lower, self.value_axis.current_upper, padding=0.0)
            plot_item.setXRange(-history_seconds, 0.0, padding=0.0)
            self.curve = plot_item.plot([], [], pen=pg.mkPen(self.general_spec.color, width=2.2))

            layout.addWidget(header, 0)
            layout.addWidget(self.plot, 1)
            self._render_current_view()

        def set_selected_joint(self, joint_index: int | None) -> None:
            self.selected_joint_index = int(joint_index) if joint_index is not None else None
            self._render_current_view()

        def ingest_snapshot(self, snapshot: DemoSnapshot) -> None:
            self.times.append(float(snapshot.time))
            self.general_values.append(float(self.general_value_getter(snapshot)))
            joint_values = np.asarray(self.joint_value_getter(snapshot), dtype=float)
            for joint_index in range(min(self.joint_count, joint_values.size)):
                self.joint_values[joint_index].append(float(joint_values[joint_index]))

            while self.times and (snapshot.time - self.times[0]) > self.history_seconds:
                self.times.popleft()
                self.general_values.popleft()
                for history in self.joint_values:
                    if history:
                        history.popleft()

            self._render_current_view()

        def _render_current_view(self) -> None:
            if self.selected_joint_index is None:
                label = self.general_spec.title
                tooltip_text = f"{self.general_spec.description}\n{self.general_spec.summary}"
                values = self.general_values
            else:
                joint_number = int(self.selected_joint_index)
                joint_idx = max(0, min(joint_number - 1, self.joint_count - 1))
                label = f"Joint {joint_number} {self.joint_spec.title}"
                tooltip_text = f"{self.joint_spec.description}\n{self.joint_spec.summary}"
                values = self.joint_values[joint_idx]

            self._tooltip_text = tooltip_text
            self.title_label.setFullText(label)

            if not self.times or not values:
                self.curve.setData([], [])
                self.value_label.setText("0.00")
                lower, upper = self.value_axis.current_lower, self.value_axis.current_upper
                self.plot.setYRange(lower, upper, padding=0.0)
                self.plot.setXRange(-self.history_seconds, 0.0, padding=0.0)
                return

            current_time = float(self.times[-1])
            xs = np.asarray(self.times, dtype=float) - current_time
            ys = np.asarray(values, dtype=float)
            self.curve.setData(xs, ys)
            self.plot.setXRange(-self.history_seconds, 0.0, padding=0.0)
            visible_min = float(np.min(ys))
            visible_max = float(np.max(ys))
            lower, upper = self.value_axis.update_band(visible_min, visible_max)
            self.plot.setYRange(lower, upper, padding=0.0)
            self.value_label.setText(f"{float(ys[-1]):.2f}")

        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
            if watched in (self.title_label, self.info_label):
                if event.type() in (QtCore.QEvent.Type.Enter, QtCore.QEvent.Type.HoverEnter):
                    self._tooltip_anchor = watched  # type: ignore[assignment]
                    self._tooltip_timer.start(self._tooltip_delay_ms)
                elif event.type() in (QtCore.QEvent.Type.MouseButtonRelease,):
                    mouse_event = event  # type: ignore[assignment]
                    if isinstance(mouse_event, QtGui.QMouseEvent) and mouse_event.button() == QtCore.Qt.MouseButton.LeftButton:
                        self._tooltip_anchor = watched  # type: ignore[assignment]
                        self._tooltip_timer.stop()
                        self._show_tooltip(QtGui.QCursor.pos(), watched)
                elif event.type() in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.Hide):
                    if self._tooltip_anchor is watched:
                        self._tooltip_timer.stop()
                        self._tooltip_anchor = None
                    QtWidgets.QToolTip.hideText()
            return super().eventFilter(watched, event)

        def _show_hover_tooltip(self) -> None:
            if self._tooltip_anchor is None or not self._tooltip_anchor.underMouse():
                return
            self._show_tooltip(QtGui.QCursor.pos(), self._tooltip_anchor)

        def _show_tooltip(self, global_pos: QtCore.QPoint, anchor: QtWidgets.QWidget) -> None:
            QtWidgets.QToolTip.showText(global_pos, self._tooltip_text, anchor)


    class SimulationCard(QtWidgets.QFrame):
        def __init__(self, arm: Arm7DOFDH, on_joint_selected: Callable[[int], None]) -> None:
            super().__init__()
            self.setObjectName("SimulationCard")
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            self.arm = arm
            self._on_joint_selected = on_joint_selected
            self._joint_chips: list[JointSelectorChip] = []
            self.world_scale = max(1.0, float(self.arm.reach))

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)

            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QVBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(8)

            title_row = QtWidgets.QWidget()
            title_row_layout = QtWidgets.QHBoxLayout(title_row)
            title_row_layout.setContentsMargins(0, 0, 0, 0)
            title_row_layout.setSpacing(10)

            self.title_label = QtWidgets.QLabel("Robot arm")
            self.title_label.setObjectName("SimulationTitle")
            self.dof_badge = QtWidgets.QLabel("7 DOF")
            self.dof_badge.setObjectName("SimulationBadge")

            title_row_layout.addWidget(self.title_label, 0)
            title_row_layout.addStretch(1)
            title_row_layout.addWidget(self.dof_badge, 0, QtCore.Qt.AlignmentFlag.AlignRight)

            chip_row = QtWidgets.QWidget()
            chip_row_layout = QtWidgets.QHBoxLayout(chip_row)
            chip_row_layout.setContentsMargins(0, 0, 0, 0)
            chip_row_layout.setSpacing(6)
            chip_row_layout.addWidget(QtWidgets.QLabel("Joint origins:"))
            for joint_index, link in enumerate(self.arm.links, start=1):
                chip = JointSelectorChip(joint_index, f"J{joint_index}")
                chip.setObjectName("SimulationChip")
                chip.setToolTip(self._format_joint_tooltip(joint_index, link))
                chip.clicked.connect(self._on_joint_selected)
                self._joint_chips.append(chip)
                chip_row_layout.addWidget(chip)
            chip_row_layout.addStretch(1)

            header_layout.addWidget(title_row)
            header_layout.addWidget(chip_row)
            layout.addWidget(header, 0)

            self.view = gl.GLViewWidget()
            self.view.setObjectName("SimulationView")
            self.view.setBackgroundColor(WINDOW_BACKGROUND)
            camera_distance = self.world_scale * 2.8
            self.view.opts["distance"] = camera_distance
            self.view.setCameraPosition(distance=camera_distance, elevation=22.0, azimuth=35.0)

            self._grid = gl.GLGridItem()
            grid_size = self.world_scale * 2.5
            self._grid.setSize(grid_size, grid_size)
            self._grid.setSpacing(self.world_scale / 8.0, self.world_scale / 8.0)
            self._grid.translate(0, 0, 0)
            self.view.addItem(self._grid)

            self._plane = self._create_xy_plane(size=self.world_scale * 2.2)
            self.view.addItem(self._plane)

            axis_length = self.world_scale * 0.55
            self._x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]], dtype=float), color=(1.0, 0.4, 0.35, 1.0), width=2, antialias=True, mode="line_strip")
            self._y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]], dtype=float), color=(0.2, 0.9, 0.5, 1.0), width=2, antialias=True, mode="line_strip")
            self._z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]], dtype=float), color=(0.2, 0.5, 1.0, 1.0), width=2, antialias=True, mode="line_strip")
            self.view.addItem(self._x_axis)
            self.view.addItem(self._y_axis)
            self.view.addItem(self._z_axis)

            self._arm_line = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=float), color=(1.0, 1.0, 1.0, 1.0), width=3, antialias=True, mode="line_strip")
            self._trace_line = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=float), color=(0.3, 0.65, 1.0, 1.0), width=2, antialias=True, mode="line_strip")
            self._joint_markers = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=float), color=(0.96, 0.8, 0.2, 1.0), size=11, pxMode=True)
            self._inactive_targets = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=float), color=(0.35, 0.55, 1.0, 1.0), size=8, pxMode=True)
            self._active_target = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=float), color=(1.0, 0.27, 0.23, 1.0), size=8, pxMode=True)

            self.view.addItem(self._arm_line)
            self.view.addItem(self._trace_line)
            self.view.addItem(self._joint_markers)
            self.view.addItem(self._inactive_targets)
            self.view.addItem(self._active_target)
            layout.addWidget(self.view, 1)

        def set_selected_joint(self, joint_index: int | None) -> None:
            for chip in self._joint_chips:
                is_selected = joint_index is not None and chip.joint_index == int(joint_index)
                chip.setProperty("selected", is_selected)
                chip.style().unpolish(chip)
                chip.style().polish(chip)
                chip.update()

        def _create_xy_plane(self, size: float) -> gl.GLMeshItem:
            half = size / 2.0
            verts = np.array(
                [
                    [-half, -half, 0.0],
                    [half, -half, 0.0],
                    [half, half, 0.0],
                    [-half, half, 0.0],
                ],
                dtype=float,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
            mesh = gl.MeshData(vertexes=verts, faces=faces)
            return gl.GLMeshItem(
                meshdata=mesh,
                color=(0.23, 0.23, 0.25, 0.18),
                smooth=False,
                shader="shaded",
                drawFaces=True,
                drawEdges=False,
                glOptions="translucent",
            )

        def update_snapshot(self, snapshot: DemoSnapshot) -> None:
            self._arm_line.setData(pos=np.asarray(snapshot.arm_points, dtype=float), color=(1.0, 1.0, 1.0, 1.0), width=3, antialias=True, mode="line_strip")
            self._trace_line.setData(pos=np.asarray(snapshot.trace_points, dtype=float), color=(0.3, 0.65, 1.0, 1.0), width=2, antialias=True, mode="line_strip")
            self._joint_markers.setData(pos=np.asarray(snapshot.arm_points[1:], dtype=float), color=(0.96, 0.8, 0.2, 1.0), size=11, pxMode=True)

            inactive = np.asarray(snapshot.inactive_targets, dtype=float) if snapshot.inactive_targets is not None else np.zeros((0, 3), dtype=float)
            self._inactive_targets.setData(pos=inactive, color=(0.35, 0.55, 1.0, 1.0), size=8, pxMode=True)

            if snapshot.active_target is not None:
                active = np.asarray(snapshot.active_target, dtype=float)[None, :]
                self._active_target.setData(pos=active, color=(1.0, 0.27, 0.23, 1.0), size=8, pxMode=True)
                self._active_target.setVisible(bool(snapshot.active_target_visible))
            else:
                self._active_target.setVisible(False)

            self.view.update()

        def _format_joint_tooltip(self, joint_index: int, link: DHLink) -> str:
            theta_offset_deg = math.degrees(link.theta_offset)
            theta_text = f"q{joint_index}" if abs(theta_offset_deg) < 1e-9 else f"q{joint_index} + {theta_offset_deg:+.1f}°"
            return (
                f"J{joint_index} DH parameters\n"
                f"a = {link.a:.3f} m, α = {math.degrees(link.alpha):+.1f}°, "
                f"d = {link.d:.3f} m, θ = {theta_text}"
            )


    class DashboardWindow(QtWidgets.QMainWindow):
        def __init__(self, session: BaseDemoSession) -> None:
            super().__init__()
            self.session = session
            self.selected_joint_index: int | None = None
            joint_getters: tuple[Callable[[DemoSnapshot], np.ndarray], ...] = (
                lambda snapshot: snapshot.joint_positions,
                lambda snapshot: snapshot.joint_velocities,
                lambda snapshot: snapshot.joint_torques,
                lambda snapshot: snapshot.joint_powers,
            )
            self.metric_cards = [
                MetricCard(
                    general_spec=general_spec,
                    joint_spec=joint_spec,
                    general_value_getter=lambda snapshot, key=general_spec.key: float(snapshot.metrics[key]),
                    joint_value_getter=joint_getter,
                    joint_count=self.session.arm.dof,
                    history_seconds=15.0,
                )
                for general_spec, joint_spec, joint_getter in zip(
                    session.metric_specs,
                    session.joint_metric_specs,
                    joint_getters,
                    strict=True,
                )
            ]
            self.sim_card = SimulationCard(self.session.arm, self._set_selected_joint)

            self.setWindowTitle(session.mode_title)
            self.setMinimumSize(1400, 900)

            root = QtWidgets.QWidget()
            root.setObjectName("DashboardRoot")
            self.setCentralWidget(root)

            outer = QtWidgets.QHBoxLayout(root)
            outer.setContentsMargins(16, 16, 16, 16)
            outer.setSpacing(16)

            left_panel = QtWidgets.QWidget()
            left_panel.setObjectName("LeftPanel")
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(16)
            for card in self.metric_cards:
                left_layout.addWidget(card, 1)

            outer.addWidget(left_panel, 4)
            outer.addWidget(self.sim_card, 6)
            outer.setStretch(0, 4)
            outer.setStretch(1, 6)

            self._apply_window_style()
            self._set_selected_joint(None)
            initial_snapshot = self.session.snapshot()
            self.sim_card.update_snapshot(initial_snapshot)
            self._ingest_snapshot(initial_snapshot)

            self._timer = QtCore.QTimer(self)
            self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._timer.timeout.connect(self._tick)
            self._timer.start(max(1, int(self.session.dt * 1000)))

        def _apply_window_style(self) -> None:
            self.setStyleSheet(
                f"""
                QMainWindow {{
                    background: {WINDOW_BACKGROUND};
                }}
                QWidget#DashboardRoot {{
                    background: {WINDOW_BACKGROUND};
                    color: {TEXT_COLOR};
                }}
                QFrame#MetricCard, QFrame#SimulationCard {{
                    background: {CARD_BACKGROUND};
                    border: 1px solid {GRID_COLOR};
                    border-radius: 18px;
                }}
                QWidget#LeftPanel {{
                    background: transparent;
                }}
                QLabel#SimulationTitle {{
                    color: {TEXT_COLOR};
                    font-size: 16px;
                }}
                QLabel#SimulationBadge {{
                    color: {WINDOW_BACKGROUND};
                    background: {ACCENT_CYAN};
                    border-radius: 10px;
                    padding: 4px 10px;
                    font-weight: 600;
                }}
                QLabel#SimulationChip {{
                    color: {TEXT_COLOR};
                    background: rgba(10, 132, 255, 0.15);
                    border: 1px solid rgba(10, 132, 255, 0.35);
                    border-radius: 10px;
                    padding: 2px 8px;
                }}
                QLabel#SimulationChip[selected="true"] {{
                    background: rgba(96, 168, 255, 0.35);
                    border: 1px solid rgba(96, 168, 255, 0.85);
                }}
                QLabel#MetricTitle, QLabel#MetricValue {{
                    color: {TEXT_COLOR};
                }}
                QLabel#MetricInfo {{
                    color: {WINDOW_BACKGROUND};
                    background: {ACCENT_CYAN};
                    border-radius: 9px;
                    font-size: 12px;
                    font-weight: 700;
                }}
                QToolTip {{
                    color: #ffffff;
                    background-color: #000000;
                    border: 1px solid #ffffff;
                    padding: 6px;
                }}
                QLabel#MetricTitle {{
                    background: transparent;
                }}
                QLabel#MetricValue {{
                    background: transparent;
                }}
                """
            )

        def _set_selected_joint(self, joint_index: int | None) -> None:
            if joint_index is None:
                next_selection: int | None = None
            elif self.selected_joint_index == joint_index:
                next_selection = None
            else:
                next_selection = int(joint_index)
            self.selected_joint_index = next_selection
            self.sim_card.set_selected_joint(self.selected_joint_index)
            for card in self.metric_cards:
                card.set_selected_joint(self.selected_joint_index)

        def _ingest_snapshot(self, snapshot: DemoSnapshot) -> None:
            for card in self.metric_cards:
                card.ingest_snapshot(snapshot)

        def _tick(self) -> None:
            snapshot = self.session.step(self.session.dt)
            self.sim_card.update_snapshot(snapshot)
            self._ingest_snapshot(snapshot)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
            self._timer.stop()
            super().closeEvent(event)


def _ensure_qt_ready() -> tuple[Path, Path]:
    issues = collect_runtime_issues()
    if issues:
        raise DemoStartupError(format_runtime_issues(issues))

    return configure_qt_plugin_paths()


def _create_application() -> QtWidgets.QApplication:
    plugin_root, _ = _ensure_qt_ready()
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([sys.argv[0]])
    QtCore.QCoreApplication.setLibraryPaths([str(plugin_root)])
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(WINDOW_BACKGROUND))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(TEXT_COLOR))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(PLOT_BACKGROUND))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(CARD_BACKGROUND))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(TEXT_COLOR))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(CARD_BACKGROUND))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(TEXT_COLOR))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(ACCENT_BLUE))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(TEXT_COLOR))
    app.setPalette(palette)
    return app


def _validate_display_backend(app: QtWidgets.QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        raise DemoStartupError(
            "Qt could not find an active display backend. Run this demo from a desktop "
            "session with a working display, then rerun `python3 src/robot_arm_3d_demo.py --mode cartesian` "
            "or `python3 src/robot_arm_3d_demo.py --mode joint`."
        )

    geometry = screen.availableGeometry()
    if geometry.width() <= 0 or geometry.height() <= 0:
        raise DemoStartupError(
            "Qt reported an unusable display geometry. Fix the graphics session, then rerun "
            "`python3 src/robot_arm_3d_demo.py --mode cartesian` or "
            "`python3 src/robot_arm_3d_demo.py --mode joint`."
        )


def _validate_opengl_backend() -> None:
    surface_format = QtGui.QSurfaceFormat()
    surface_format.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)

    surface = QtGui.QOffscreenSurface()
    surface.setFormat(surface_format)
    surface.create()
    if not surface.isValid():
        raise DemoStartupError(
            "Qt could not create an offscreen OpenGL surface. Install a working OpenGL "
            "stack, then rerun `python3 src/robot_arm_3d_demo.py --mode cartesian` or "
            "`python3 src/robot_arm_3d_demo.py --mode joint`."
        )

    context = QtGui.QOpenGLContext()
    context.setFormat(surface_format)
    if not context.create() or not context.isValid():
        raise DemoStartupError(
            "Qt could not create an OpenGL context. Install a working OpenGL stack, then "
            "rerun `python3 src/robot_arm_3d_demo.py --mode cartesian` or "
            "`python3 src/robot_arm_3d_demo.py --mode joint`."
        )


def _run_demo(session: BaseDemoSession) -> int:
    app = _create_application()
    _validate_display_backend(app)
    _validate_opengl_backend()
    window = DashboardWindow(session)

    screen = app.primaryScreen()
    if screen is not None:
        geometry = screen.availableGeometry()
        window.resize(max(1400, int(geometry.width() * 0.95)), max(900, int(geometry.height() * 0.92)))

    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


def run_cartesian_demo() -> int:
    return _run_demo(CartesianTrackingSession())


def run_joint_demo() -> int:
    return _run_demo(JointStepSession())
