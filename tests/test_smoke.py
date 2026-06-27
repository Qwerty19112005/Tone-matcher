"""Smoke tests — confirm the package imports and the CLI parser builds.

These do NOT touch audio, plugins, or torch-heavy paths; they just verify the project is
wired together so CI/local `pytest` passes from a clean install.
"""

from __future__ import annotations


def test_package_imports():
    import tonematcher

    assert tonematcher.__version__


def test_subpackages_import():
    # Each stage package should at least import cleanly.
    import importlib

    for name in ("hosting", "metrics", "optimize", "profile", "nominate", "data"):
        importlib.import_module(f"tonematcher.{name}")


def test_cli_parser_builds():
    from tonematcher.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert hasattr(args, "command")
