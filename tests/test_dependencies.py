from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXACT_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _exact_pins(requirements: list[str]) -> dict[str, str]:
    pins = {}
    for requirement in requirements:
        match = _EXACT_PIN.match(requirement.strip())
        if match:
            pins[_canonical_name(match.group(1))] = match.group(2)
    return pins


def test_dependency_lock_matches_direct_pins() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    requirements = [
        *project["dependencies"],
        *(
            requirement
            for group in project["optional-dependencies"].values()
            for requirement in group
        ),
    ]
    direct = _exact_pins(requirements)
    assert len(direct) == len(requirements), "Every direct dependency must use an exact pin"

    locked = _exact_pins(
        (PROJECT_ROOT / "requirements.lock").read_text(encoding="utf-8-sig").splitlines()
    )
    stale = {
        name: {"pyproject": version, "requirements.lock": locked.get(name)}
        for name, version in sorted(direct.items())
        if locked.get(name) != version
    }
    assert not stale, f"requirements.lock is out of date with pyproject.toml: {stale}"
