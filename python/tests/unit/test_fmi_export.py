"""FMU packaging and failure contracts, without a licensed oracle."""

import json
import zipfile
from pathlib import Path

import pyraichu
import pytest
from pyraichu import fmi


def _model() -> pyraichu.Model:
    return pyraichu.load_model({
        "name": "exported",
        "components": [{
            "name": "plant",
            "attributes": [{
                "name": "signal", "kind": "float",
                "init": {"kind": "float", "value": 2.0},
            }],
        }],
    })


def test_export_writes_description_resource_and_runtime(tmp_path: Path) -> None:
    destination = fmi.export(_model(), tmp_path / "plant.fmu", outputs=["plant.signal"])
    host = fmi._host_platform()
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "modelDescription.xml",
            "resources/raichu-model.json",
            f"binaries/{host}/{fmi._LIBRARY_NAMES[host]}",
        }
        assert b'causality="output"' in archive.read("modelDescription.xml")
        assert json.loads(archive.read("resources/raichu-model.json"))["export"]["outputs"] == [
            "plant.signal"
        ]


def test_export_adds_extra_platform_library(tmp_path: Path) -> None:
    host = fmi._host_platform()
    other = next(name for name in fmi._LIBRARY_NAMES if name != host)
    source = tmp_path / "extra-runtime"
    source.write_bytes(b"test library")
    destination = fmi.export(_model(), tmp_path / "plant.fmu", extra_libraries={other: source})
    with zipfile.ZipFile(destination) as archive:
        assert archive.read(f"binaries/{other}/{fmi._LIBRARY_NAMES[other]}") == b"test library"


def test_invalid_manifest_does_not_create_archive(tmp_path: Path) -> None:
    destination = tmp_path / "plant.fmu"
    with pytest.raises(pyraichu.ModelError, match="unknown attribute.*plant.missing"):
        fmi.export(_model(), destination, outputs=["plant.missing"])
    assert not destination.exists()


def test_string_attribute_is_refused(tmp_path: Path) -> None:
    document = json.loads(_model().json)
    document["components"][0]["attributes"][0].update(
        kind="string", init={"kind": "string", "value": "text"}
    )
    model = pyraichu.Model(json=json.dumps(document))
    destination = tmp_path / "plant.fmu"
    with pytest.raises(pyraichu.ModelError, match="string|String"):
        fmi.export(model, destination, outputs=["plant.signal"])
    assert not destination.exists()


def test_missing_host_runtime_is_typed_and_names_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fmi, "__file__", str(tmp_path / "fmi.py"))
    destination = tmp_path / "plant.fmu"
    with pytest.raises(fmi.MissingRuntimeError, match=fmi._host_platform()):
        fmi.export(_model(), destination)
    assert not destination.exists()
