"""Unit tests of the oracle-environment build script.

`scripts/build_oracle_env.py` rebuilds the cross-validation oracle from
three pinned ingredients (a muscadet tag, an interpreter, the reference
engine) recorded in a committed config. The heavy steps it drives (clone,
venv, install) are integration work verified by running the script; what
belongs here is the decision logic a rebuild rests on: reading and
validating the config, locating and versioning the closed reference
engine, and refusing a build whose engine is below the declared floor
with a message that names what was found.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "build_oracle_env.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_oracle_env", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def oenv():
    return _load_script()


class TestLoadConfig:
    def test_valid_config_loads(self, oenv, tmp_path):
        config = tmp_path / "oracle-env.json"
        config.write_text(
            '{"muscadet_tag": "5.6.0", "python": "3.10",'
            ' "pycatshoo_floor": "1.3.8.0"}'
        )
        loaded = oenv.load_config(config)
        assert loaded["muscadet_tag"] == "5.6.0"
        assert loaded["python"] == "3.10"

    def test_missing_ingredient_is_refused(self, oenv, tmp_path):
        config = tmp_path / "oracle-env.json"
        config.write_text('{"muscadet_tag": "5.6.0"}')
        with pytest.raises(oenv.OracleEnvError) as excinfo:
            oenv.load_config(config)
        assert "python" in str(excinfo.value)


class TestProbeEngineVersion:
    def test_version_is_read_from_the_shared_object(self, oenv, tmp_path):
        so = tmp_path / "Pycatshoo.so"
        # A binary is mostly non-printable bytes; the version sits in a
        # printable run, as it does in the real object.
        so.write_bytes(b"\x7fELF\x02\x01\x00garbage\x00" b"1.3.8.0\x00trailing")
        assert oenv.probe_engine_version(so) == "1.3.8.0"

    def test_ambiguous_versions_are_refused(self, oenv, tmp_path):
        so = tmp_path / "Pycatshoo.so"
        so.write_bytes(b"1.3.8.0\x00later rebuild\x001.4.1.0\x00")
        with pytest.raises(oenv.OracleEnvError) as excinfo:
            oenv.probe_engine_version(so)
        assert "1.3.8.0" in str(excinfo.value)
        assert "1.4.1.0" in str(excinfo.value)

    def test_absent_version_is_reported(self, oenv, tmp_path):
        so = tmp_path / "Pycatshoo.so"
        so.write_bytes(b"\x7fELF no version string here")
        assert oenv.probe_engine_version(so) is None


class TestVersionFloor:
    def test_equal_version_is_not_below_floor(self, oenv):
        assert not oenv.version_below_floor("1.3.8.0", "1.3.8.0")

    def test_older_version_is_below_floor(self, oenv):
        assert oenv.version_below_floor("1.2.4.3", "1.3.8.0")
        assert oenv.version_below_floor("1.3.7.2", "1.3.8.0")

    def test_newer_version_is_not_below_floor(self, oenv):
        assert not oenv.version_below_floor("1.4.1.0", "1.3.8.0")

    def test_malformed_floor_is_refused(self, oenv):
        with pytest.raises(oenv.OracleEnvError):
            oenv.version_below_floor("1.3.8.0", "one.point.four")


class TestLocateEngine:
    def _make_tree(self, root: Path, version: str) -> Path:
        lib = root / "Core" / "lib"
        lib.mkdir(parents=True)
        payload = b"\x7fELF\x00" + version.encode() + b"\x00"
        (lib / "Pycatshoo.so").write_bytes(payload)
        return root

    def test_home_argument_wins(self, oenv, tmp_path):
        home = self._make_tree(tmp_path / "engine-a", "1.4.1.0")
        self._make_tree(tmp_path / "engine-b", "1.3.8.0")
        found = oenv.engine_candidates(
            str(home), [str(tmp_path / "engine-*")]
        )
        assert found == [home / "Core" / "lib"]

    def test_search_glob_orders_newest_first(self, oenv, tmp_path):
        older = self._make_tree(tmp_path / "pycatshoo_1.2.4.3", "1.2.4.3")
        newer = self._make_tree(tmp_path / "pycatshoo_1.4.1.0", "1.4.1.0")
        found = oenv.engine_candidates(None, [str(tmp_path / "pycatshoo*")])
        assert found == [newer / "Core" / "lib", older / "Core" / "lib"]

    def test_missing_engine_is_refused_with_what_was_searched(
        self, oenv, tmp_path
    ):
        with pytest.raises(oenv.OracleEnvError) as excinfo:
            oenv.engine_candidates(None, [str(tmp_path / "nowhere*")])
        assert "nowhere*" in str(excinfo.value)

    def test_engine_below_floor_is_refused_naming_the_version(
        self, oenv, tmp_path
    ):
        self._make_tree(tmp_path / "pycatshoo_old", "1.2.4.3")
        module_dir = oenv.engine_candidates(
            None, [str(tmp_path / "pycatshoo*")]
        )[0]
        version = oenv.probe_engine_version(module_dir / "Pycatshoo.so")
        with pytest.raises(oenv.OracleEnvError) as excinfo:
            oenv.check_engine_floor(module_dir, version, "1.3.8.0")
        assert "1.2.4.3" in str(excinfo.value)


class TestPlan:
    def test_plan_names_dedicated_paths_outside_the_repo(self, oenv, tmp_path):
        config = {
            "muscadet_tag": "5.6.0",
            "python": "3.10",
            "pycatshoo_floor": "1.3.8.0",
        }
        plan = oenv.plan(config, tmp_path)
        assert plan["venv"] == tmp_path / "venv-muscadet-5.6.0"
        assert plan["clone"] == tmp_path / "muscadet"
        assert plan["record"] == tmp_path / "oracle-env.json"
