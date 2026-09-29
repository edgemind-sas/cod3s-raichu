use std::collections::HashSet;

use raichu_expr::Value;
use raichu_model::{AttrKind, Model};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::MODEL_IDENTIFIER;

/// Explicit FMU surface. Names are qualified `component.attribute` names.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportManifest {
    /// Attributes writable at communication points.
    #[serde(default)]
    pub inputs: Vec<String>,
    /// Attributes readable by the importer.
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Attribute values tunable before initialization.
    #[serde(default)]
    pub parameters: Vec<String>,
}

/// One scalar exported variable.
#[derive(Clone, Debug)]
pub struct Variable {
    /// Qualified attribute name or fixed runtime parameter name.
    pub name: String,
    /// FMI value reference.
    pub value_reference: u32,
    /// FMI scalar element tag.
    pub kind: &'static str,
    /// FMI causality.
    pub causality: &'static str,
    /// Initial scalar value.
    pub start: String,
}

/// Invalid export document or manifest.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// JSON syntax or shape is invalid.
    #[error("export JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The RAICHU model document is invalid.
    #[error("RAICHU model: {0}")]
    Model(String),
    /// The manifest is inconsistent with the model.
    #[error("export manifest: {0}")]
    Manifest(String),
}

/// Attach a manifest to a sealed RAICHU model document.
pub fn prepare_document(
    model_json: &str,
    manifest: &ExportManifest,
) -> Result<String, ExportError> {
    let mut value: serde_json::Value = serde_json::from_str(model_json)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| ExportError::Manifest("model document must be an object".into()))?;
    object.insert("export".into(), serde_json::to_value(manifest)?);
    let result = serde_json::to_string_pretty(&value)?;
    inspect(&result)?;
    Ok(result)
}

/// Generate the FMI 3 model description for the exact resource bytes.
pub fn model_description(document: &str) -> Result<String, ExportError> {
    let (model, _, variables) = inspect(document)?;
    let token = token(document.as_bytes());
    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<fmiModelDescription fmiVersion=\"3.0\" modelName=\"{}\" instantiationToken=\"{}\" generationTool=\"RAICHU\" variableNamingConvention=\"structured\">\n  <CoSimulation modelIdentifier=\"{}\" canHandleVariableCommunicationStepSize=\"true\" canGetAndSetFMUState=\"true\" canSerializeFMUState=\"false\"/>\n  <ModelVariables>\n",
        escape(&model.name), token, MODEL_IDENTIFIER
    );
    for variable in &variables {
        let variability = if variable.causality == "parameter" {
            "fixed"
        } else if variable.kind == "Float64" {
            "continuous"
        } else {
            "discrete"
        };
        let initial = if variable.causality == "output" {
            " initial=\"calculated\""
        } else if variable.causality == "parameter" {
            " initial=\"exact\""
        } else {
            ""
        };
        let start = if matches!(variable.causality, "output" | "independent") {
            String::new()
        } else {
            format!(" start=\"{}\"", escape(&variable.start))
        };
        xml.push_str(&format!("    <{} name=\"{}\" valueReference=\"{}\" causality=\"{}\" variability=\"{}\"{}{} />\n", variable.kind, escape(&variable.name), variable.value_reference, variable.causality, variability, initial, start));
    }
    xml.push_str("  </ModelVariables>\n  <ModelStructure>\n");
    for variable in variables.iter().filter(|v| v.causality == "output") {
        xml.push_str(&format!(
            "    <Output valueReference=\"{}\"/>\n",
            variable.value_reference
        ));
    }
    for variable in variables.iter().filter(|v| v.causality == "output") {
        xml.push_str(&format!(
            "    <InitialUnknown valueReference=\"{}\"/>\n",
            variable.value_reference
        ));
    }
    xml.push_str("  </ModelStructure>\n</fmiModelDescription>\n");
    Ok(xml)
}

pub(crate) fn token(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

pub(crate) fn inspect(
    document: &str,
) -> Result<(Model, ExportManifest, Vec<Variable>), ExportError> {
    let value: serde_json::Value = serde_json::from_str(document)?;
    let manifest: ExportManifest = serde_json::from_value(
        value
            .get("export")
            .cloned()
            .ok_or_else(|| ExportError::Manifest("missing export manifest".into()))?,
    )?;
    let model =
        Model::from_json(document).map_err(|error| ExportError::Model(error.to_string()))?;
    model
        .validate()
        .map_err(|error| ExportError::Model(error.to_string()))?;
    if !model.fmu_units.is_empty() {
        return Err(ExportError::Manifest(
            "nested FMU imports cannot be exported".into(),
        ));
    }
    let mut variables = vec![
        Variable {
            name: "seed".into(),
            value_reference: 1,
            kind: "UInt64",
            causality: "parameter",
            start: "0".into(),
        },
        Variable {
            name: "rng_stream".into(),
            value_reference: 2,
            kind: "UInt64",
            causality: "parameter",
            start: "0".into(),
        },
        Variable {
            name: "time".into(),
            value_reference: 3,
            kind: "Float64",
            causality: "independent",
            start: "0".into(),
        },
    ];
    let mut names = HashSet::from([
        "seed".to_owned(),
        "rng_stream".to_owned(),
        "time".to_owned(),
    ]);
    for (causality, entries) in [
        ("input", &manifest.inputs),
        ("output", &manifest.outputs),
        ("parameter", &manifest.parameters),
    ] {
        for name in entries {
            if !names.insert(name.clone()) {
                return Err(ExportError::Manifest(format!(
                    "duplicate variable `{name}`"
                )));
            }
            let mut parts = name.split('.');
            let component_name = parts.next().unwrap_or_default();
            let attribute_name = parts.next().unwrap_or_default();
            if component_name.is_empty() || attribute_name.is_empty() || parts.next().is_some() {
                return Err(ExportError::Manifest(format!(
                    "invalid qualified attribute `{name}`"
                )));
            }
            let attribute = model
                .components
                .iter()
                .find(|c| c.name == component_name)
                .and_then(|c| c.attributes.iter().find(|a| a.name == attribute_name))
                .ok_or_else(|| ExportError::Manifest(format!("unknown attribute `{name}`")))?;
            let kind = match attribute.kind {
                AttrKind::Bool => "Boolean",
                AttrKind::Int => "Int64",
                AttrKind::Float => "Float64",
            };
            let start = match attribute.init {
                Value::Bool(v) => v.to_string(),
                Value::Int(v) => v.to_string(),
                Value::Float(v) => v.to_string(),
            };
            variables.push(Variable {
                name: name.clone(),
                value_reference: variables.len() as u32 + 1,
                kind,
                causality,
                start,
            });
        }
    }
    Ok((model, manifest, variables))
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
