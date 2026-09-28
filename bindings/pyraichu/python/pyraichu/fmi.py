"""Export a RAICHU model as an FMI 3 Co-Simulation archive."""

from __future__ import annotations

import json
import platform
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import Model
from ._pyraichu import prepare_fmu_export

_LIBRARY_NAMES = {
    "x86_64-linux": "raichu_fmu.so",
    "aarch64-linux": "raichu_fmu.so",
    "aarch64-darwin": "raichu_fmu.dylib",
    "x86_64-windows": "raichu_fmu.dll",
}


class MissingRuntimeError(FileNotFoundError):
    """The installed pyraichu wheel has no FMU runtime for a platform."""

    def __init__(self, platform_tuple: str):
        self.platform_tuple = platform_tuple
        super().__init__(f"FMU runtime for {platform_tuple} is absent from pyraichu")


def _host_platform() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    operating_system = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(system)
    return f"{architecture}-{operating_system or system.lower()}"


def export(
    model: Model,
    path: str | Path,
    *,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    parameters: Sequence[str] = (),
    extra_libraries: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write an FMI 3 Co-Simulation FMU using packaged and supplied runtimes.

    ``inputs``, ``outputs`` and ``parameters`` contain qualified
    ``component.attribute`` names. The host runtime comes from the wheel;
    ``extra_libraries`` maps additional FMI 3 platform tuples to native files.
    The model and manifest are validated before the destination is created.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a validated pyraichu.Model")
    manifest = {
        "inputs": list(inputs),
        "outputs": list(outputs),
        "parameters": list(parameters),
    }
    document, description = prepare_fmu_export(model.json, json.dumps(manifest))

    host = _host_platform()
    if host not in _LIBRARY_NAMES:
        raise MissingRuntimeError(host)
    runtime = Path(__file__).parent / "runtime" / host / _LIBRARY_NAMES[host]
    if not runtime.is_file():
        raise MissingRuntimeError(host)
    libraries = {host: runtime}
    for platform_tuple, source in (extra_libraries or {}).items():
        if platform_tuple not in _LIBRARY_NAMES:
            raise ValueError(f"unknown FMI 3 platform tuple {platform_tuple!r}")
        if platform_tuple == host:
            raise ValueError(f"extra library duplicates host platform {host!r}")
        library = Path(source)
        if not library.is_file():
            raise FileNotFoundError(f"FMU runtime for {platform_tuple}: {library}")
        libraries[platform_tuple] = library

    destination = Path(path)
    if destination.suffix.lower() != ".fmu":
        raise ValueError("FMU destination must have a .fmu suffix")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("modelDescription.xml", description)
        archive.writestr("resources/raichu-model.json", document)
        for platform_tuple, library in libraries.items():
            archive.write(library, f"binaries/{platform_tuple}/{_LIBRARY_NAMES[platform_tuple]}")
    return destination
