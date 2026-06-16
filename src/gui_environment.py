from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Iterable

SUPPORTED_PYTHON_VERSIONS: tuple[tuple[int, int], ...] = ((3, 12), (3, 13))


def _format_version(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _supported_python_message() -> str:
    versions = ", ".join(_format_version(version) for version in SUPPORTED_PYTHON_VERSIONS)
    current = _format_version(sys.version_info[:2])
    return f"Python {current} is not supported for this app. Use Python 3.13, or 3.12 if 3.13 is unavailable. Supported versions: {versions}."


def _import_or_message(module_name: str, label: str | None = None) -> str | None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        name = label or module_name
        return f"{name} import failed: {exc}"
    return None


def qt_plugin_paths() -> tuple[Path, Path]:
    from PySide6 import QtCore

    plugin_root = Path(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath))
    platform_root = plugin_root / "platforms"
    return plugin_root, platform_root


def configure_qt_plugin_paths() -> tuple[Path, Path]:
    plugin_root, platform_root = qt_plugin_paths()
    os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_root)

    from PySide6 import QtCore

    QtCore.QCoreApplication.setLibraryPaths([str(plugin_root)])
    return plugin_root, platform_root


def collect_runtime_issues() -> list[str]:
    issues: list[str] = []

    if sys.version_info[:2] not in SUPPORTED_PYTHON_VERSIONS:
        issues.append(_supported_python_message())

    pyside6_ok = True
    for module_name, label in (
        ("numpy", "numpy"),
        ("PySide6", "PySide6"),
        ("pyqtgraph", "pyqtgraph"),
        ("OpenGL", "PyOpenGL"),
        ("OpenGL.GL", "OpenGL.GL"),
        ("pyqtgraph.opengl", "pyqtgraph.opengl"),
    ):
        message = _import_or_message(module_name, label)
        if message:
            issues.append(message)
            if module_name == "PySide6":
                pyside6_ok = False

    if pyside6_ok:
        try:
            plugin_root, platform_root = qt_plugin_paths()
        except Exception as exc:  # pragma: no cover - environment dependent
            issues.append(f"Qt plugin paths could not be resolved from PySide6: {exc}")
        else:
            if not plugin_root.is_dir():
                issues.append(f"Qt plugin root does not exist: {plugin_root}")
            if not platform_root.is_dir():
                issues.append(f"Qt platform plugin directory does not exist: {platform_root}")
            cocoa = platform_root / "libqcocoa.dylib"
            if not cocoa.is_file():
                issues.append(f"Qt Cocoa platform plugin is missing: {cocoa}")

    return issues


def format_runtime_issues(issues: Iterable[str]) -> str:
    issues = list(issues)
    if not issues:
        return "Environment check passed."
    return "Environment check failed:\n" + "\n".join(f"- {issue}" for issue in issues)
