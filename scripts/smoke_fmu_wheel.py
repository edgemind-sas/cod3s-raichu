"""Check that an installed release wheel can export and execute an FMU."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pyraichu
from fmpy import simulate_fmu
from fmpy.validation import validate_fmu
from pyraichu import fmi


def main() -> None:
    model = pyraichu.load_model(
        {
            "name": "wheel-smoke",
            "components": [
                {
                    "name": "plant",
                    "attributes": [
                        {"name": "x", "kind": "float", "init": {"kind": "float", "value": 0.0}}
                    ],
                    "equations": [
                        {
                            "target": "x",
                            "kind": "ode",
                            "expr": {
                                "op": "const",
                                "value": {"kind": "float", "value": 1.0},
                            },
                        }
                    ],
                }
            ],
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        path = fmi.export(model, Path(directory) / "wheel-smoke.fmu", outputs=["plant.x"])
        with zipfile.ZipFile(path) as archive:
            binaries = [name for name in archive.namelist() if name.startswith("binaries/")]
            assert len(binaries) == 1, binaries
            assert binaries[0].startswith(f"binaries/{fmi._host_platform()}/"), binaries
        assert validate_fmu(str(path)) == []
        result = simulate_fmu(
            str(path),
            start_time=0.0,
            stop_time=1.0,
            step_size=0.5,
            output_interval=0.5,
            output=["plant.x"],
        )
        assert abs(float(result[-1]["plant.x"]) - 1.0) < 1e-7


if __name__ == "__main__":
    main()
