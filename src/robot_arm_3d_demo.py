from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="7DOF robotic arm demo runner")
    parser.add_argument(
        "--mode",
        choices=("cartesian", "joint"),
        default="cartesian",
        help="Choose the main Cartesian demo or the joint-space validation demo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if __package__:
        from .robot_arm_qt_dashboard import run_cartesian_demo, run_joint_demo
        from .robot_arm_qt_dashboard import DemoStartupError
    else:
        from robot_arm_qt_dashboard import DemoStartupError, run_cartesian_demo, run_joint_demo

    try:
        if args.mode == "joint":
            return run_joint_demo()
        else:
            return run_cartesian_demo()
    except DemoStartupError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
