"""Build the generic FMU runtime and stage it for maturin package data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

TARGETS = {
    "x86_64-unknown-linux-gnu": ("x86_64-linux", "libraichu_fmu.so", "raichu_fmu.so"),
    "aarch64-unknown-linux-gnu": ("aarch64-linux", "libraichu_fmu.so", "raichu_fmu.so"),
    "aarch64-apple-darwin": ("aarch64-darwin", "libraichu_fmu.dylib", "raichu_fmu.dylib"),
    "x86_64-pc-windows-msvc": ("x86_64-windows", "raichu_fmu.dll", "raichu_fmu.dll"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = ["cargo", "build", "-p", "raichu-fmu", "--target", arguments.target]
    if arguments.release:
        command.append("--release")
    subprocess.run(command, cwd=root, check=True)
    platform_tuple, source_name, destination_name = TARGETS[arguments.target]
    profile = "release" if arguments.release else "debug"
    source = root / "target" / arguments.target / profile / source_name
    destination = (
        root / "bindings" / "pyraichu" / "python" / "pyraichu"
        / "runtime" / platform_tuple / destination_name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(destination)


if __name__ == "__main__":
    main()
