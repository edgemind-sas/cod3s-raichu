#!/usr/bin/env python3
"""Build the cross-validation oracle environment from pinned ingredients.

The cross-validation suite (``python/tests/validation/``) adjudicates
every parity claim this repository makes, so the oracle it runs against
must never depend on a path recorded in a session, a shell profile or a
teammate's memory. This script rebuilds that oracle from a committed
config naming three pinned ingredients:

1. a **muscadet tag**, cloned into a dedicated checkout (never the
   shared sibling checkout, whose moving would disturb every other
   consumer of it) and installed **non-editable**, so no meta-path
   finder can quietly re-point the oracle at another tree;
2. an **interpreter** pin (a ``pythonX.Y`` major.minor), resolved
   through ``uv`` when available;
3. the **reference engine** itself, a closed distribution no script can
   fetch: it is *located and verified* rather than installed, refused
   below the declared floor with a message naming what was found, and
   recorded with its resolved path and version.

Everything the run later states about the oracle is written to a record
file beside the environment, and the suite re-states the three
ingredients in its own artefact on every run.

Usage:

    scripts/build_oracle_env.py [--config PATH] [--pycatshoo-home DIR]
                                [--workspace DIR] [--muscadet-from SRC]

Stdlib only; ``uv`` is preferred when present and ``python -m venv`` +
``pip`` is the fallback.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "python" / "tests" / "validation" / "oracle-env.json"
DEFAULT_WORKSPACE = Path("~/.cache/raichu-oracle")
#: Name of the record file, stated once: the validation runner resolves
#: the built environment through this same name.
RECORD_NAME = "oracle-env.json"
DEFAULT_MUSCADET_ORIGIN = "git@github.com:edgemind-sas/muscadet.git"

_SIBLING_MUSCADET = REPO_ROOT.parent / "modules" / "muscadet"

_VERSION_IN_SO = re.compile(r"^\d+\.\d+(?:\.\d+){1,3}$")
_PRINTABLE_RUN = re.compile(rb"[ -~]{4,}")


class OracleEnvError(RuntimeError):
    """A pinned ingredient could not be resolved or verified."""


def load_config(path: Path) -> dict:
    """Read the committed oracle config and refuse an incomplete one.

    The three ingredients that decide a verdict (muscadet tag,
    interpreter pin, reference-engine floor) must all be present; the
    search globs and the workspace directory are machine conveniences
    and may be absent.
    """
    data = json.loads(path.read_text())
    missing = [
        key for key in ("muscadet_tag", "python", "pycatshoo_floor") if key not in data
    ]
    if missing:
        raise OracleEnvError(
            f"oracle config {path} is missing pinned ingredient(s): "
            f"{', '.join(missing)}"
        )
    return data


def probe_engine_version(so: Path) -> str | None:
    """The version string embedded in the engine's ``Pycatshoo.so``.

    The reference engine ships no version file its import path exposes,
    but the shared object carries its distribution version as a
    printable string. Exactly one distinct version-like string reads as
    the version; none reads as unknown; several are refused rather than
    guessed at.
    """
    candidates = set()
    data = so.read_bytes()
    for match in _PRINTABLE_RUN.finditer(data):
        text = match.group().decode().strip()
        if _VERSION_IN_SO.fullmatch(text):
            candidates.add(text)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise OracleEnvError(
            f"{so} carries several version-like strings "
            f"({', '.join(sorted(candidates))}); the engine version cannot "
            "be certified from this distribution"
        )
    return candidates.pop()


def parse_version(version: str) -> tuple[int, ...]:
    """`"1.3.8.0"` as a comparable tuple of integers."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        raise OracleEnvError(f"malformed version string: {version!r}") from None


def version_below_floor(version: str, floor: str) -> bool:
    """Whether `version` sorts strictly below the declared `floor`."""
    return parse_version(version) < parse_version(floor)


def engine_candidates(home: str | None, search_globs: list[str]) -> list[Path]:
    """Directories holding an importable ``Pycatshoo.so``, newest first.

    An explicit home (CLI flag or ``RAICHU_PYCATSHOO_HOME``) wins and
    yields the single candidate it names (the distribution root or its
    ``Core/lib`` directory directly). Without one, the config's search
    globs are tried newest-name-first, so a machine carrying several
    builds prefers its newest and falls back down the list. A candidate
    needs a ``Pycatshoo.so`` FILE: a distribution's build tree may carry
    a directory of that name that is not the module. Refusal names what
    was searched.
    """
    found: list[Path] = []
    if home:
        for candidate in (Path(home) / "Core" / "lib", Path(home)):
            if (candidate / "Pycatshoo.so").is_file():
                return [candidate]
        raise OracleEnvError(
            f"--pycatshoo-home / RAICHU_PYCATSHOO_HOME={home} holds no "
            "Pycatshoo.so "
            "(looked for <home>/Core/lib/Pycatshoo.so and <home>/Pycatshoo.so)"
        )
    seen: set[str] = set()
    for pattern in reversed(search_globs):
        expanded = os.path.expanduser(pattern)
        for candidate in sorted(glob_module.glob(expanded), reverse=True):
            module_dir = Path(candidate) / "Core" / "lib"
            if str(module_dir) in seen:
                continue
            seen.add(str(module_dir))
            if (module_dir / "Pycatshoo.so").is_file():
                found.append(module_dir)
    if not found:
        raise OracleEnvError(
            "no reference-engine distribution found; searched: "
            + ", ".join(search_globs)
            + " (set RAICHU_PYCATSHOO_HOME or --pycatshoo-home to the install root)"
        )
    return found


def hermetic_env() -> dict:
    """The caller's environment without its ambient ``PYTHONPATH``.

    The oracle must be certified as what it resolves ALONE: an ambient
    ``PYTHONPATH`` shadows the wired engine through sys.path ordering
    and would let a probe approve a module the environment never loads
    on its own. One definition, shared with the validation runner, so
    the engine the build certifies and the engine the suite probes can
    never drift apart.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _uv_available() -> bool:
    """Whether uv is actually executable, not merely present on PATH."""
    return (
        subprocess.run(["uv", "--version"], capture_output=True, check=False).returncode
        == 0
    )


def select_engine(
    venv_python: Path, candidates: list[Path]
) -> tuple[Path, str | None]:
    """The first candidate whose module the built interpreter can import.

    A version string alone does not make an engine usable: a module
    built against another Python's shared library resolves its version
    fine and still fails to load. Each candidate is wired into the venv
    (a ``.pth`` file, the mechanism a venv itself uses) and actually
    imported, in a hermetic environment; the first that loads is
    selected, its wiring stays, and the refusals of the others are named
    if none survives.
    """
    failures: list[str] = []
    env = hermetic_env()
    site_packages = sorted(
        venv_python.parent.parent.glob("lib/python*/site-packages")
    )
    if not site_packages:
        raise OracleEnvError(f"no site-packages directory found under {venv_python}")
    for module_dir in candidates:
        pth = site_packages[-1] / "pycatshoo-engine.pth"
        pth.write_text(str(module_dir) + "\n")
        probe = subprocess.run(
            # -I keeps the venv's own wiring (site-packages and .pth
            # files) while dropping the calling process's cwd from
            # sys.path and ignoring PYTHONPATH: without it, a cwd
            # carrying a module of the probed name would answer the
            # import and a non-importable candidate could be selected.
            [str(venv_python), "-I", "-c", "import Pycatshoo"],
            capture_output=True,
            check=False,
            text=True,
            env=env,
        )
        if probe.returncode == 0:
            version = probe_engine_version(module_dir / "Pycatshoo.so")
            return module_dir, version
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else ""
        failures.append(f"{module_dir}: {detail}")
        pth.unlink()
    raise OracleEnvError(
        "no reference-engine distribution imports under the built "
        "interpreter:\n  " + "\n  ".join(failures)
    )


def check_engine_floor(module_dir: Path, version: str | None, floor: str) -> None:
    """Refuse an engine that cannot be certified against the floor."""
    if version is None:
        raise OracleEnvError(
            f"no version string found in {module_dir}; "
            f"cannot certify it against the declared floor {floor}"
        )
    if version_below_floor(version, floor):
        raise OracleEnvError(
            f"reference engine at {module_dir} reports version {version}, "
            f"below the declared floor {floor}: rebuild against a newer "
            "distribution before trusting the bench"
        )


def resolve_interpreter(pin: str) -> tuple[Path, str]:
    """The CPython interpreter for the pinned major.minor, with its version.

    The resolved ingredient is the interpreter's BASE, never a virtual
    environment: `uv python find` run inside a project can return that
    project's own .venv, and an oracle venv recorded as descending from
    a disposable venv would state an ingredient that outlives nothing.
    """
    probe = subprocess.run(
        ["uv", "python", "find", pin], capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0:
        candidate = Path(probe.stdout.strip().splitlines()[-1])
    else:
        which = shutil.which(f"python{pin}")
        if which is None:
            raise OracleEnvError(
                f"no interpreter found for pin {pin}: tried `uv python find` "
                f"and `python{pin}` on PATH"
            )
        candidate = Path(which)
    base_probe = subprocess.run(
        [str(candidate), "-c",
         "import sys\nprint(sys.base_prefix)\nprint('Python %d.%d.%d' % sys.version_info[:3])\n"],
        capture_output=True,
        check=False,
        text=True,
    )
    if base_probe.returncode != 0:
        detail = (
            base_probe.stderr.strip().splitlines()[-1]
            if base_probe.stderr.strip()
            else f"exit {base_probe.returncode}"
        )
        raise OracleEnvError(f"resolved interpreter {candidate} does not run: {detail}")
    base, reported = base_probe.stdout.strip().splitlines()[-2:]
    path = Path(base) / "bin" / "python"
    return path, reported


def plan(config: dict, workspace: Path) -> dict:
    """The dedicated paths a build occupies, derived from the config.

    The clone, the venv and the record all live under the workspace
    directory and never inside the repository: the venv is a disposable
    build artefact and the clone must not be confused with any checkout
    another consumer owns.
    """
    tag = config["muscadet_tag"]
    return {
        "clone": workspace / "muscadet",
        "venv": workspace / f"venv-muscadet-{tag}",
        "record": workspace / RECORD_NAME,
    }


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    shown = " ".join(command)
    result = subprocess.run(command, check=False, **kwargs)
    if result.returncode != 0:
        raise OracleEnvError(f"command failed ({result.returncode}): {shown}")
    return result


def prepare_clone(source: str, clone: Path, tag: str) -> tuple[str, str]:
    """A dedicated muscadet checkout at the pinned tag; returns (tag, sha).

    The clone never touches the tree it clones from: a local source is
    cloned with ``--no-hardlinks`` so the two trees share no object file.
    An existing clone is reused only when clean, and moved to the tag.
    The resolved commit SHA rides with the tag everywhere, because a tag
    is a mutable ref and the bench's ingredient claim must stay
    verifiable if it ever moves.
    """
    if clone.exists():
        dirty = _run(
            ["git", "-C", str(clone), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if dirty.stdout.strip():
            raise OracleEnvError(
                f"dedicated muscadet clone {clone} is dirty; clean or remove "
                "it before rebuilding"
            )
        _run(["git", "-C", str(clone), "fetch", "origin", "--tags", "--force"])
    else:
        command = ["git", "clone"]
        if Path(source).expanduser().exists():
            command.append("--no-hardlinks")
        command += [source, str(clone)]
        _run(command)
    _run(["git", "-C", str(clone), "checkout", "--force", tag])
    sha = _run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tag, sha


def build_venv(interpreter: Path, venv: Path) -> None:
    """A fresh virtual environment from the pinned interpreter.

    An existing venv is removed first: the environment is a disposable
    product of the pinned ingredients, and a rebuild that layered onto a
    previous one could inherit state no ingredient accounts for.
    """
    if venv.exists():
        shutil.rmtree(venv)
    if _uv_available():
        _run(["uv", "venv", "--python", str(interpreter), str(venv)])
    else:
        _run([str(interpreter), "-m", "venv", str(venv)])


def install_muscadet(clone: Path, venv: Path) -> None:
    """Install muscadet non-editable from the dedicated clone.

    Installing from the clone rather than installing editable is what
    keeps the oracle honest: an editable install's meta-path finder takes
    precedence over ``sys.path``, so an editable oracle would silently
    answer capability probes from whatever tree the finder recorded
    instead of the pinned one.
    """
    python = venv / "bin" / "python"
    if _uv_available():
        _run(["uv", "pip", "install", "--python", str(python), str(clone)])
    else:
        _run([str(python), "-m", "ensurepip"])
        _run([str(python), "-m", "pip", "install", str(clone)])


def write_record(
    paths: dict,
    config: dict,
    muscadet_sha: str,
    interpreter: tuple[Path, str],
    engine_module_dir: Path,
    engine_version: str | None,
) -> None:
    """State the three resolved ingredients beside the environment.

    Written atomically: the runner reads this file at collection time,
    so a truncated write would fail the whole suite as a decode error
    rather than the build failing alone.
    """
    record_text = json.dumps(
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "muscadet": {
                "tag": config["muscadet_tag"],
                "commit": muscadet_sha,
                "clone": str(paths["clone"]),
            },
            "interpreter": {
                "pin": config["python"],
                "path": str(interpreter[0]),
                "version": interpreter[1],
            },
            "pycatshoo": {
                "module_dir": str(engine_module_dir),
                "version": engine_version,
                "floor": config["pycatshoo_floor"],
            },
            "venv_python": str(paths["venv"] / "bin" / "python"),
        },
        indent=2,
    ) + "\n"
    tmp = paths["record"].with_name(paths["record"].name + ".tmp")
    tmp.write_text(record_text)
    tmp.replace(paths["record"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pycatshoo-home",
        default=os.environ.get("RAICHU_PYCATSHOO_HOME"),
        help="reference-engine install root (default: RAICHU_PYCATSHOO_HOME, "
        "then the config's search globs)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="directory for the dedicated clone, the venv and the record "
        "(default: the config's workspace_dir, then ~/.cache/raichu-oracle)",
    )
    parser.add_argument(
        "--muscadet-from",
        default=None,
        help="where to clone muscadet from (default: the sibling checkout "
        "when present, else the EdgeMind origin)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    workspace = (
        args.workspace
        or Path(
            config.get("workspace_dir", str(DEFAULT_WORKSPACE))
        ).expanduser()
    )
    workspace.mkdir(parents=True, exist_ok=True)
    paths = plan(config, workspace)

    source = args.muscadet_from or (
        str(_SIBLING_MUSCADET)
        if (_SIBLING_MUSCADET / "muscadet" / "engine.py").exists()
        else DEFAULT_MUSCADET_ORIGIN
    )
    tag, sha = prepare_clone(source, paths["clone"], config["muscadet_tag"])
    interpreter = resolve_interpreter(config["python"])
    build_venv(interpreter[0], paths["venv"])
    engine_module_dir, engine_version = select_engine(
        paths["venv"] / "bin" / "python",
        engine_candidates(
            args.pycatshoo_home, config.get("pycatshoo_search_globs", [])
        ),
    )
    check_engine_floor(engine_module_dir, engine_version, config["pycatshoo_floor"])
    install_muscadet(paths["clone"], paths["venv"])
    write_record(paths, config, sha, interpreter, engine_module_dir, engine_version)

    print("oracle environment rebuilt:")
    print(f"  muscadet    {tag} ({sha[:12]}) from {source}")
    print(f"  interpreter {interpreter[1]} at {interpreter[0]}")
    print(
        f"  pycatshoo   {engine_version} at {engine_module_dir} "
        f"(floor {config['pycatshoo_floor']})"
    )
    print(f"  record      {paths['record']}")
    print(f'  export RAICHU_ORACLE_PYTHON={paths["venv"] / "bin" / "python"}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
