//! Integration tests against the vendored Reference FMUs.
#![allow(clippy::expect_used, clippy::panic, clippy::err_expect)]
use std::path::PathBuf;

use raichu_fmi::{Archive, FmiError, FmiVersion, Value};
use sha2::{Digest, Sha256};

fn fixture(version: &str, model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/reference-fmus")
        .join(version)
        .join(format!("{model}.fmu"))
}

fn runtime_versions() -> &'static [&'static str] {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        eprintln!("Reference FMUs v0.0.41 FMI 2 runtime excluded on aarch64-darwin: its darwin64 binary is x86_64 only");
        &["3.0"]
    } else {
        &["2.0", "3.0"]
    }
}

#[test]
fn reads_both_versions_without_loading_a_library() {
    for (version, expected) in [("2.0", FmiVersion::V2), ("3.0", FmiVersion::V3)] {
        let archive = Archive::open(fixture(version, "Dahlquist")).expect("reference FMU opens");
        assert_eq!(archive.description().fmi_version, expected);
        assert_eq!(archive.description().model_name, "Dahlquist");
        assert_eq!(
            archive.description().generation_tool_version.as_deref(),
            Some("0.0.41")
        );
        assert!(archive.description().variable("x").is_some());
        let expected_hash =
            Sha256::digest(std::fs::read(fixture(version, "Dahlquist")).expect("FMU bytes"));
        assert_eq!(archive.sha256(), format!("sha256:{expected_hash:x}"));
    }
}

#[test]
fn dahlquist_steps_on_both_versions() {
    for &version in runtime_versions() {
        let archive = Archive::open(fixture(version, "Dahlquist")).expect("reference FMU opens");
        let mut instance = archive
            .instantiate("dahlquist", 0.0)
            .expect("FMU instantiates");
        let x = archive.description().variable("x").expect("state output");
        let golden_name = format!(
            "dahlquist-fmi{}-fmusim-0.12.0.csv",
            if version == "2.0" { 2 } else { 3 }
        );
        let golden = std::fs::read_to_string(
            fixture(version, "Dahlquist")
                .parent()
                .expect("fixture directory")
                .parent()
                .expect("fixture root")
                .join(golden_name),
        )
        .expect("fmusim golden");
        let points: Vec<(f64, f64)> = golden
            .lines()
            .skip(1)
            .map(|line| {
                let (time, value) = line.split_once(',').expect("CSV pair");
                (time.parse().expect("time"), value.parse().expect("value"))
            })
            .collect();
        assert_eq!(
            instance.get_value(x).expect("initial output"),
            Value::Float(points[0].1)
        );
        for k in 0..10 {
            instance.do_step(points[k].0, 0.1).expect("step succeeds");
            let Value::Float(actual) = instance.get_value(x).expect("point output") else {
                panic!("not Float64")
            };
            let expected = points[k + 1].1;
            assert!(
                (actual - expected).abs() < 1e-12,
                "{version} point {k}: {actual} versus {expected}"
            );
        }
        instance.terminate().expect("terminate succeeds");
    }
}

#[test]
fn feedthrough_exchanges_three_scalar_kinds() {
    for &version in runtime_versions() {
        let archive = Archive::open(fixture(version, "Feedthrough")).expect("reference FMU opens");
        let mut instance = archive
            .instantiate("feedthrough", 0.0)
            .expect("FMU instantiates");
        let (float_in, float_out, int_in, int_out, bool_in, bool_out) = (
            "Float64_continuous_input",
            "Float64_continuous_output",
            "Int32_input",
            "Int32_output",
            "Boolean_input",
            "Boolean_output",
        );
        let variable = |name: &str| {
            archive
                .description()
                .variable(name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        instance
            .set_value(variable(float_in), Value::Float(2.5))
            .expect("set Float64");
        instance
            .set_value(variable(int_in), Value::Int(42))
            .expect("set integer");
        instance
            .set_value(variable(bool_in), Value::Bool(true))
            .expect("set Boolean");
        instance.do_step(0.0, 0.1).expect("step succeeds");
        assert_eq!(
            instance
                .get_value(variable(float_out))
                .expect("get Float64"),
            Value::Float(2.5)
        );
        assert_eq!(
            instance.get_value(variable(int_out)).expect("get integer"),
            Value::Int(42)
        );
        assert_eq!(
            instance.get_value(variable(bool_out)).expect("get Boolean"),
            Value::Bool(true)
        );
    }
}

#[test]
fn missing_binary_names_the_unit_and_platform() {
    let archive = Archive::open(fixture("3.0", "Dahlquist")).expect("reference FMU opens");
    let path = archive.path().join("binaries");
    std::fs::remove_dir_all(path).expect("remove test binary");
    let error = archive
        .instantiate("plant", 0.0)
        .err()
        .expect("must refuse");
    assert!(matches!(error, FmiError::MissingBinary { .. }));
    assert!(error.to_string().contains("plant"));
}

#[test]
fn serialized_state_restores_a_trajectory() {
    for &version in runtime_versions() {
        let archive = Archive::open(fixture(version, "Dahlquist")).expect("reference FMU opens");
        let mut instance = archive
            .instantiate("dahlquist", 0.0)
            .expect("FMU instantiates");
        let x = archive.description().variable("x").expect("state output");
        let saved = instance.get_state().expect("serialize state");
        instance.do_step(0.0, 0.1).expect("step");
        assert_ne!(
            instance.get_value(x).expect("value after step"),
            Value::Float(1.0)
        );
        instance.set_state(&saved).expect("restore state");
        assert_eq!(
            instance.get_value(x).expect("restored value"),
            Value::Float(1.0)
        );
    }
}

#[test]
fn refuses_path_traversal_symlinks_and_oversized_entries() {
    use std::io::Write;
    let cases = [
        ("../evil", None, 1_u64),
        ("escape", Some(0o120777), 1),
        ("huge", None, 512 * 1024 * 1024 + 1),
    ];
    for (name, mode, size) in cases {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("malicious.fmu");
        let file = std::fs::File::create(&path).expect("create archive");
        let mut writer = zip::ZipWriter::new(file);
        let mut options = zip::write::SimpleFileOptions::default();
        if let Some(mode) = mode {
            options = options.unix_permissions(mode);
        }
        writer.start_file(name, options).expect("start entry");
        if size == 1 {
            writer.write_all(b"x").expect("write entry");
        }
        writer.finish().expect("finish archive");
        if size > 1 || mode.is_some() {
            // A sparse ZIP entry is patched with a declared decompressed size
            // so the reader can refuse it before allocating or writing bytes.
            let mut bytes = std::fs::read(&path).expect("read archive");
            let position = bytes
                .windows(4)
                .position(|w| w == b"PK\x01\x02")
                .expect("central directory");
            if size > 1 {
                bytes[position + 24..position + 28].copy_from_slice(&(size as u32).to_le_bytes());
            }
            if mode.is_some() {
                bytes[position + 38..position + 42]
                    .copy_from_slice(&(0o120777_u32 << 16).to_le_bytes());
            }
            std::fs::write(&path, bytes).expect("patch archive");
        }
        let error = Archive::open(&path).err().expect("must refuse");
        if size > 1 {
            assert!(matches!(error, FmiError::Limit { .. }), "{name}: {error}");
        } else {
            assert!(
                matches!(error, FmiError::UnsafeEntry { .. }),
                "{name}: {error}"
            );
        }
        assert!(!directory.path().join("evil").exists());
    }
}

#[test]
fn refuses_wrong_instantiation_token() {
    use std::io::{Read, Write};
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("wrong-token.fmu");
    let original = std::fs::File::open(fixture("3.0", "Dahlquist")).expect("reference FMU");
    let mut source = zip::ZipArchive::new(original).expect("source ZIP");
    let mut writer = zip::ZipWriter::new(std::fs::File::create(&path).expect("target ZIP"));
    for index in 0..source.len() {
        let mut entry = source.by_index(index).expect("entry");
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).expect("entry bytes");
        if name == "modelDescription.xml" {
            let xml = String::from_utf8(bytes).expect("XML text");
            let start =
                xml.find("instantiationToken=\"").expect("token") + "instantiationToken=\"".len();
            let end = start + xml[start..].find('"').expect("token end");
            bytes = format!("{}wrong-token{}", &xml[..start], &xml[end..]).into_bytes();
        }
        writer
            .start_file(name, zip::write::SimpleFileOptions::default())
            .expect("write entry");
        writer.write_all(&bytes).expect("write bytes");
    }
    writer.finish().expect("finish ZIP");
    let archive = Archive::open(path).expect("description is readable");
    let error = archive
        .instantiate("wrong-token-unit", 0.0)
        .err()
        .expect("must refuse");
    assert!(matches!(error, FmiError::Instantiate { .. }), "{error}");
    assert!(error.to_string().contains("wrong-token-unit"));
}

#[test]
fn refuses_model_exchange_without_co_simulation() {
    let xml = r#"<fmiModelDescription fmiVersion="3.0" modelName="OnlyME" instantiationToken="token"><ModelExchange modelIdentifier="OnlyME"/><ModelVariables/></fmiModelDescription>"#;
    let error = raichu_fmi::Description::parse(xml)
        .err()
        .expect("must refuse");
    assert!(matches!(error, FmiError::Description(_)));
    assert!(error.to_string().contains("CoSimulation"));
}

#[test]
fn refuses_model_identifiers_that_escape_the_binary_folder() {
    for identifier in ["", ".", "..", "../outside", "/tmp/outside", "C:\\outside"] {
        let xml = format!(
            "<fmiModelDescription fmiVersion=\"3.0\" modelName=\"Bad\" instantiationToken=\"token\"><CoSimulation modelIdentifier=\"{identifier}\"/><ModelVariables/></fmiModelDescription>"
        );
        let error = raichu_fmi::Description::parse(&xml)
            .err()
            .expect("unsafe identifier must be refused");
        assert!(
            matches!(error, FmiError::Description(_)),
            "{identifier}: {error}"
        );
    }
}

#[test]
fn fmi3_arrays_cannot_be_used_as_scalar_variables() {
    let xml = r#"<fmiModelDescription fmiVersion="3.0" modelName="Array" instantiationToken="token"><CoSimulation modelIdentifier="Array"/><ModelVariables><Float64 name="values" valueReference="1" causality="output"><Dimensions><Dimension start="2"/></Dimensions></Float64></ModelVariables></fmiModelDescription>"#;
    let description = raichu_fmi::Description::parse(xml).expect("description parses");
    let variable = description
        .variable("values")
        .expect("array variable exists");
    assert!(matches!(
        variable.kind,
        raichu_fmi::VariableKind::Unsupported(_)
    ));
}
