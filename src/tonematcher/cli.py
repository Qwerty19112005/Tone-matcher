"""Command-line entry point for tonematcher.

Currently a thin stub. Real subcommands (profile / nominate / match) arrive in later
phases; for now the project is driven by the scripts in ``scripts/`` (Phase 0/1).
"""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tonematcher",
        description="Match a target guitar tone using your local plugin library.",
    )
    parser.add_argument("--version", action="version", version=f"tonematcher {__version__}")
    parser.set_defaults(func=None)

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    # Placeholder subcommands — implemented in later phases.
    sub.add_parser("profile", help="(Phase 4) sample + render + embed a plugin's tone region")
    sub.add_parser("nominate", help="(Phase 4) propose candidate plugins/chains for a target")
    sub.add_parser("match", help="(Phase 1+) optimize knob settings to match a target")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    print(f"'{args.command}' is not implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
