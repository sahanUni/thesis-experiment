"""Report installed versions against requirements.txt without needing pip.

A hand-built Python may have a venv with no pip at all, and the packages that
decide whether this machine can generate official results -- numpy and mujoco
above all -- have to be checked before any compute is spent. Exits non-zero if
a pinned requirement is missing or does not match.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
PINNED = re.compile(r"^([A-Za-z0-9_.-]+)==(.+)$")


def main() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    print(f"{'package':<22}{'installed':<14}{'required':<14}status")
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PINNED.match(line)
        name = match.group(1) if match else re.split(r"[<>=!]", line, 1)[0]
        required = match.group(2) if match else line[len(name):] or "any"
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"{name:<22}{'-':<14}{required:<14}MISSING")
            problems.append(f"{name} is not installed")
            continue
        if match and installed != required:
            print(f"{name:<22}{installed:<14}{required:<14}MISMATCH")
            problems.append(f"{name} is {installed}, requirements pin {required}")
        else:
            print(f"{name:<22}{installed:<14}{required:<14}ok")

    print(f"\npython {sys.version.split()[0]}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        print("\nnumpy and mujoco decide the physics. Fix these before training:")
        print("  python -m pip install -r requirements.txt")
        sys.exit(1)
    print("\nAll pinned requirements match. Run check_parity.py next.")


if __name__ == "__main__":
    main()
