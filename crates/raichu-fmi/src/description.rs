use roxmltree::{Document, Node};

use crate::FmiError;

/// FMI standard generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FmiVersion {
    /// FMI 2.0.
    V2,
    /// FMI 3.0.
    V3,
}

/// Scalar kind used for RAICHU bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariableKind {
    /// A 64-bit floating-point variable.
    Float64,
    /// An integer variable, widened from FMI 2's 32-bit representation.
    Int64,
    /// A Boolean variable.
    Boolean,
    /// A variable kind that cannot bind to a RAICHU attribute.
    Unsupported(String),
}

/// One variable from `modelDescription.xml`.
#[derive(Clone, Debug)]
pub struct Variable {
    /// Variable name.
    pub name: String,
    /// FMI value reference.
    pub value_reference: u32,
    /// Scalar kind.
    pub kind: VariableKind,
    /// Integer width in bits for FMI 3's distinct integer entry points.
    pub integer_width: Option<u8>,
    /// FMI causality, normally `input`, `output`, or `parameter`.
    pub causality: String,
    /// FMI variability.
    pub variability: String,
    /// Initial scalar text, when declared.
    pub start: Option<String>,
}

/// Co-simulation description and capability flags; parsing does not load a library.
#[derive(Clone, Debug)]
pub struct Description {
    /// FMI standard generation.
    pub fmi_version: FmiVersion,
    /// Name of the model.
    pub model_name: String,
    /// FMI 2 GUID or FMI 3 instantiation token.
    pub token: String,
    /// Binary model identifier.
    pub model_identifier: String,
    /// Generating tool name, when declared.
    pub generation_tool: Option<String>,
    /// Generating tool version, when declared.
    pub generation_tool_version: Option<String>,
    /// Whether a step may end at a non-grid communication point.
    pub can_handle_variable_communication_step_size: bool,
    /// Whether native state get/set is available.
    pub can_get_and_set_fmu_state: bool,
    /// Whether native state can be serialized across instances.
    pub can_serialize_fmu_state: bool,
    /// Whether the FMU allows only one instance per process.
    pub can_be_instantiated_only_once_per_process: bool,
    /// Declared scalar variables.
    pub variables: Vec<Variable>,
}

impl Description {
    /// Parse an FMI 2 or 3 co-simulation description.
    pub fn parse(xml: &str) -> Result<Self, FmiError> {
        let doc = Document::parse(xml).map_err(|e| FmiError::Description(e.to_string()))?;
        let root = doc.root_element();
        if root.tag_name().name() != "fmiModelDescription" {
            return Err(FmiError::Description(
                "root is not <fmiModelDescription>".into(),
            ));
        }
        let version = required(root, "fmiVersion")?;
        let fmi_version = if version.starts_with("2.0") {
            FmiVersion::V2
        } else if version.starts_with("3.0") {
            FmiVersion::V3
        } else {
            return Err(FmiError::Description(format!(
                "FMI version `{version}` is not supported"
            )));
        };
        let cs = root
            .children()
            .find(|n| n.is_element() && n.tag_name().name() == "CoSimulation")
            .ok_or_else(|| FmiError::Description("missing <CoSimulation> interface".into()))?;
        let model_name = required(root, "modelName")?.to_owned();
        let token = required(
            root,
            if fmi_version == FmiVersion::V2 {
                "guid"
            } else {
                "instantiationToken"
            },
        )?
        .to_owned();
        let model_identifier = required(cs, "modelIdentifier")?.to_owned();
        if model_identifier.is_empty()
            || model_identifier == "."
            || model_identifier == ".."
            || model_identifier
                .chars()
                .any(|character| matches!(character, '/' | '\\' | ':' | '\0'))
        {
            return Err(FmiError::Description(format!(
                "invalid co-simulation modelIdentifier `{model_identifier}`"
            )));
        }
        let variable_root = root
            .children()
            .find(|n| n.is_element() && n.tag_name().name() == "ModelVariables")
            .ok_or_else(|| FmiError::Description("missing <ModelVariables>".into()))?;
        let mut variables = Vec::new();
        for node in variable_root.children().filter(Node::is_element) {
            let (scalar, kind_name) = if fmi_version == FmiVersion::V2 {
                if node.tag_name().name() != "ScalarVariable" {
                    continue;
                }
                let scalar = node.children().find(Node::is_element).ok_or_else(|| {
                    FmiError::Description(format!(
                        "variable `{}` has no scalar type",
                        node.attribute("name").unwrap_or("?")
                    ))
                })?;
                (scalar, scalar.tag_name().name())
            } else {
                (node, node.tag_name().name())
            };
            let kind = match kind_name {
                "Real" | "Float64" => VariableKind::Float64,
                "Integer" | "Int64" | "Int32" => VariableKind::Int64,
                "Boolean" => VariableKind::Boolean,
                other => VariableKind::Unsupported(other.to_owned()),
            };
            let kind = if fmi_version == FmiVersion::V3
                && node
                    .descendants()
                    .any(|child| child.is_element() && child.tag_name().name() == "Dimension")
            {
                VariableKind::Unsupported(format!("{kind_name} array"))
            } else {
                kind
            };
            let value_reference = required(node, "valueReference")?.parse().map_err(|_| {
                FmiError::Description(format!(
                    "variable `{}` has invalid valueReference",
                    node.attribute("name").unwrap_or("?")
                ))
            })?;
            variables.push(Variable {
                name: required(node, "name")?.to_owned(),
                value_reference,
                kind,
                integer_width: match kind_name {
                    "Integer" | "Int32" => Some(32),
                    "Int64" => Some(64),
                    _ => None,
                },
                causality: node.attribute("causality").unwrap_or("local").to_owned(),
                variability: node
                    .attribute("variability")
                    .unwrap_or("continuous")
                    .to_owned(),
                start: scalar.attribute("start").map(str::to_owned),
            });
        }
        Ok(Self {
            fmi_version,
            model_name,
            token,
            model_identifier,
            generation_tool: root.attribute("generationTool").map(str::to_owned),
            generation_tool_version: root
                .attribute("generationToolVersion")
                .map(str::to_owned)
                .or_else(|| root.attribute("generationTool").and_then(tool_version)),
            can_handle_variable_communication_step_size: boolean(
                cs,
                "canHandleVariableCommunicationStepSize",
            ),
            can_get_and_set_fmu_state: boolean(cs, "canGetAndSetFMUstate")
                || boolean(cs, "canGetAndSetFMUState"),
            can_serialize_fmu_state: boolean(cs, "canSerializeFMUstate")
                || boolean(cs, "canSerializeFMUState"),
            can_be_instantiated_only_once_per_process: boolean(
                cs,
                "canBeInstantiatedOnlyOncePerProcess",
            ),
            variables,
        })
    }

    /// Look up a variable by its declared name.
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|v| v.name == name)
    }
}

fn required<'a>(node: Node<'a, '_>, key: &str) -> Result<&'a str, FmiError> {
    node.attribute(key)
        .ok_or_else(|| FmiError::Description(format!("<{}> lacks `{key}`", node.tag_name().name())))
}

fn boolean(node: Node<'_, '_>, key: &str) -> bool {
    node.attribute(key) == Some("true")
}

fn tool_version(tool: &str) -> Option<String> {
    let start = tool.rfind("(v")? + 2;
    let end = tool[start..].find(')')? + start;
    let version = &tool[start..end];
    if version.chars().all(|c| c.is_ascii_digit() || c == '.') {
        Some(version.to_owned())
    } else {
        None
    }
}
