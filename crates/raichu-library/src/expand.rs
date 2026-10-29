//! The expansion of a system (and optionally a study) into a flat core
//! model.

use crate::document::{Library, Simulation, StudyDocument, SystemModel};
use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use crate::instantiate::{instantiate, ExpandedInstance, LibrarySet};
use crate::names::split_public;
use crate::provenance::{Origin, Provenance};
use crate::std_library::{CONTINUOUS_FLOW_FRAGMENTS, STD_UNBOUNDED_RATE};
use crate::strict::typed;
use raichu_expr::PortRef;
use raichu_model::{Connection, Indicator, Model, ModelSource, PortDir};

/// The result of an expansion.
#[derive(Debug, Clone)]
pub struct Expansion {
    /// The flat core model, ready for validation and compilation.
    pub model: Model,
    /// Where every element of `model` came from.
    pub provenance: Provenance,
    /// The study's run settings, when a study with settings was expanded.
    pub simulation: Option<Simulation>,
    /// Non-fatal observations (for example a type-wide activation on a
    /// type with no instance).
    pub warnings: Vec<String>,
}

/// Expands `system`, drawing its types from `libraries`, and applies
/// `study` when given.
///
/// The library the system names must be among `libraries` at exactly the
/// version the system names; the libraries it imports must be there too.
pub fn expand(
    libraries: &[Library],
    system: &SystemModel,
    study: Option<&StudyDocument>,
) -> Result<Expansion, ExpandError> {
    let libraries = &crate::std_library::with_std(libraries)?;
    let set = LibrarySet { libraries };
    let library = bind_library(libraries, system)?;
    for import in &library.imports {
        if !libraries.iter().any(|l| &l.name == import) {
            return Err(ExpandError::new(
                DocumentKind::Library,
                "/library/imports",
                format!(
                    "library `{}` imports `{import}`, which is not available",
                    library.name
                ),
            ));
        }
    }
    for (i, ty) in library.types.iter().enumerate() {
        if library.types[..i].iter().any(|t| t.name == ty.name)
            || library.fragments.iter().any(|f| f.name == ty.name)
        {
            return Err(ExpandError::new(
                DocumentKind::Library,
                format!("/library/types/{i}/name"),
                format!(
                    "library `{}` declares the name `{}` twice",
                    library.name, ty.name
                ),
            ));
        }
    }

    let mut instances: Vec<ExpandedInstance> = Vec::with_capacity(system.instances.len());
    for (i, instance) in system.instances.iter().enumerate() {
        let pointer = pointer_index("/system/instances", i);
        if system.instances[..i]
            .iter()
            .any(|other| other.name == instance.name)
        {
            return Err(ExpandError::new(
                DocumentKind::System,
                pointer_key(&pointer, "name"),
                format!("instance name `{}` is used twice", instance.name),
            ));
        }
        let type_index = library
            .types
            .iter()
            .position(|t| t.name == instance.type_name)
            .ok_or_else(|| {
                ExpandError::new(
                    DocumentKind::System,
                    pointer_key(&pointer, "type"),
                    format!(
                        "instance `{}`: library `{}` has no type `{}`",
                        instance.name, library.name, instance.type_name
                    ),
                )
            })?;
        instances.push(instantiate(&set, library, type_index, instance, &pointer)?);
    }

    let mut provenance = Provenance::default();
    let connections = connect(system, &instances, &mut provenance)?;
    let mut indicators = Vec::new();
    for (i, raw) in system.indicators.iter().enumerate() {
        let pointer = pointer_index("/system/indicators", i);
        let indicator: Indicator = typed(raw, DocumentKind::System, &pointer, "")?;
        provenance.insert(
            format!("indicator:{}", indicator.name),
            Origin::system(pointer),
        );
        indicators.push(indicator);
    }
    let mut targets = Vec::new();
    for (i, target) in system.targets.iter().enumerate() {
        provenance.insert(
            format!("target:{}", target.name),
            Origin::system(pointer_index("/system/targets", i)),
        );
        targets.push(target.clone());
    }

    let mut expansion = Expansion {
        model: Model {
            name: system.name.clone(),
            components: Vec::new(),
            connections,
            interface_connections: Vec::new(),
            indicators,
            targets,
            programs: Vec::new(),
            evaluation_order: None,
            unbounded_rate: None,
            fmu_units: Vec::new(),
            source: Some(ModelSource {
                library: library.name.clone(),
                library_version: library.version.clone(),
                system: system.name.clone(),
                study: study.map(|s| s.name.clone()),
                std_version: library
                    .imports
                    .iter()
                    .any(|i| i == crate::std_library::STD)
                    .then(|| crate::std_library::STD_VERSION.to_owned()),
            }),
        },
        provenance,
        simulation: None,
        warnings: Vec::new(),
    };
    let groups = match study {
        Some(study) => crate::study::apply(study, system, &mut instances, &mut expansion)?,
        None => Vec::new(),
    };
    for instance in instances {
        for (key, origin) in instance.provenance {
            expansion.provenance.insert(key, origin);
        }
        expansion.model.components.push(instance.component);
    }
    // After the instances: a group's combinations rank after its members'
    // effect steps at a shared date, as PyCATSHOO's mode components after
    // the targets carrying their mirrors.
    expansion.model.components.extend(groups);
    expansion.model.unbounded_rate = unbounded_rate(system, &expansion.provenance)?;
    // The sweep order is derived, never authored (KTD14). A model the
    // derivation cannot read is structurally invalid, and validation below
    // reports it with its origin.
    if let Ok(order) = crate::order::derive_evaluation_order(&expansion.model) {
        expansion.model.evaluation_order = order;
    }
    expansion
        .model
        .validate()
        .map_err(|e| restate(&e.to_string(), &expansion.provenance))?;
    Ok(expansion)
}

/// Restates an error the core raised on the flat model with the origin of
/// the element it names.
fn restate(message: &str, provenance: &Provenance) -> ExpandError {
    match provenance.locate(message) {
        Some((_, origin)) => {
            let instance = origin
                .instance
                .as_ref()
                .map(|i| format!(" for instance `{i}`"))
                .unwrap_or_default();
            ExpandError::new(
                origin.document,
                origin.pointer.clone(),
                format!("{message}; this comes from {}{instance}", origin.describe()),
            )
        }
        None => ExpandError::new(
            DocumentKind::System,
            "",
            format!("the expanded model is invalid: {message}"),
        ),
    }
}

/// The `unbounded_rate` of the expanded model.
///
/// An instance applying a standard continuous flow fragment makes the model
/// a continuous network, which carries the magnitude the standard fragments
/// write for "unbounded" ([`STD_UNBOUNDED_RATE`]), as the muscadet plugin
/// declares it on every system with a continuous flow: without it the
/// engine would integrate that magnitude as a rate instead of refusing it
/// by name. Such a system may state only that magnitude; another system
/// declares the one it states, or none.
fn unbounded_rate(
    system: &SystemModel,
    provenance: &Provenance,
) -> Result<Option<f64>, ExpandError> {
    let continuous = provenance.elements.values().any(|origin| {
        origin
            .fragment
            .as_deref()
            .is_some_and(|f| CONTINUOUS_FLOW_FRAGMENTS.contains(&f))
    });
    match (continuous, system.unbounded_rate) {
        (false, stated) => Ok(stated),
        (true, None) => Ok(Some(STD_UNBOUNDED_RATE)),
        #[allow(clippy::float_cmp)] // the same constant, not a computed value
        (true, Some(stated)) if stated == STD_UNBOUNDED_RATE => Ok(Some(stated)),
        (true, Some(stated)) => Err(ExpandError::new(
            DocumentKind::System,
            "/system/unbounded_rate",
            format!(
                "system `{}` states unbounded_rate {stated:e}, but it applies the standard \
                 continuous flow fragments, whose unbounded magnitude is {STD_UNBOUNDED_RATE:e}: \
                 state that one or none",
                system.name
            ),
        )),
    }
}

fn bind_library<'a>(
    libraries: &'a [Library],
    system: &SystemModel,
) -> Result<&'a Library, ExpandError> {
    let wanted = &system.library;
    let candidates: Vec<&Library> = libraries.iter().filter(|l| l.name == wanted.name).collect();
    if let Some(library) = candidates.iter().find(|l| l.version == wanted.version) {
        return Ok(library);
    }
    let message = match candidates.first() {
        Some(other) => format!(
            "system `{}` binds library `{}` at version `{}`, but the library supplied is at version `{}`",
            system.name, wanted.name, wanted.version, other.version
        ),
        None => format!(
            "system `{}` binds library `{}`, which is not among the libraries supplied",
            system.name, wanted.name
        ),
    };
    Err(ExpandError::new(
        DocumentKind::System,
        "/system/library",
        message,
    ))
}

fn connect(
    system: &SystemModel,
    instances: &[ExpandedInstance],
    provenance: &mut Provenance,
) -> Result<Vec<Connection>, ExpandError> {
    let mut out = Vec::with_capacity(system.connections.len());
    for (i, connection) in system.connections.iter().enumerate() {
        let pointer = pointer_index("/system/connections", i);
        let from = endpoint(
            instances,
            &connection.from,
            PortDir::Out,
            &pointer_key(&pointer, "from"),
        )?;
        let to = endpoint(
            instances,
            &connection.to,
            PortDir::In,
            &pointer_key(&pointer, "to"),
        )?;
        let key = connection
            .name
            .clone()
            .unwrap_or_else(|| format!("{}->{}", connection.from, connection.to));
        provenance.insert(format!("connection:{key}"), Origin::system(pointer));
        out.push(Connection {
            name: connection.name.clone(),
            from,
            to,
        });
    }
    Ok(out)
}

fn endpoint(
    instances: &[ExpandedInstance],
    written: &str,
    dir: PortDir,
    pointer: &str,
) -> Result<PortRef, ExpandError> {
    let (instance_name, port_name) = split_public(written).ok_or_else(|| {
        ExpandError::new(
            DocumentKind::System,
            pointer,
            format!("`{written}` is not a port reference; write `Instance.port`"),
        )
    })?;
    let instance = instances
        .iter()
        .find(|i| i.component.name == instance_name)
        .ok_or_else(|| {
            ExpandError::new(
                DocumentKind::System,
                pointer,
                format!("no instance `{instance_name}`"),
            )
        })?;
    let port =
        instance
            .component
            .ports
            .iter()
            .find(|p| p.name == port_name)
            .ok_or_else(|| {
                let known: Vec<String> = instance
                    .component
                    .ports
                    .iter()
                    .map(|p| format!("`{}`", p.name))
                    .collect();
                ExpandError::new(
                    DocumentKind::System,
                    pointer,
                    format!(
                    "instance `{instance_name}` (type `{}`) has no port `{port_name}` (ports: {})",
                    instance.type_name,
                    if known.is_empty() { "none".to_owned() } else { known.join(", ") }
                ),
                )
            })?;
    if port.dir != dir {
        let (want, have) = match dir {
            PortDir::Out => ("an out port", "an in port"),
            PortDir::In => ("an in port", "an out port"),
        };
        return Err(ExpandError::new(
            DocumentKind::System,
            pointer,
            format!("a connection goes from an out port to an in port; `{written}` is {have}, {want} is expected here"),
        ));
    }
    Ok(PortRef {
        component: instance_name.to_owned(),
        port: port_name.to_owned(),
    })
}
