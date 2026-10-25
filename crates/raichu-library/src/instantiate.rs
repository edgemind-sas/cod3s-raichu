//! Instantiation: from a type, its applied fragments and an instance's
//! parameter values to one core component.

use crate::document::{Effect, Instance, Library, Nature, Parameter, TypeDef, Use};
use crate::error::{pointer_index, pointer_key, DocumentKind, ExpandError};
use crate::provenance::Origin;
use crate::strict::{located, typed};
use crate::template::{bind, substitute, Ctx, Env};
use raichu_expr::Expr;
use raichu_model::{
    Allocation, Attribute, Automaton, Component, Equation, Interface, Port, SensitiveFunction,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The body of a type or fragment once its templates are resolved: the
/// parts of a core component, without its name.
#[derive(Debug, Clone, Default, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct BodyParts {
    #[serde(default)]
    attributes: Vec<Attribute>,
    #[serde(default)]
    ports: Vec<Port>,
    #[serde(default)]
    interfaces: Vec<Interface>,
    #[serde(default)]
    automata: Vec<Automaton>,
    #[serde(default)]
    sensitive_functions: Vec<SensitiveFunction>,
    #[serde(default)]
    equations: Vec<Equation>,
    #[serde(default)]
    allocations: Vec<Allocation>,
}

/// How the effects of several active modes on one attribute combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CombineRule {
    /// Boolean: false while at least one active mode cuts.
    AnyCuts,
    /// Boolean: true while at least one active mode activates.
    AnyActivates,
    /// Numeric: the smallest value held by an active mode.
    Min,
    /// Numeric: the largest value held by an active mode.
    Max,
}

/// An attribute a mode may target, with its combination rule.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct Targetable {
    pub attribute: String,
    pub rule: CombineRule,
}

/// One entry of a type's mode catalogue, templates resolved.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct CatalogueMode {
    pub name: String,
    pub nature: Nature,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "Option<crate::schema::Condition>")
    )]
    pub condition: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub description: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<Value>,
}

/// One expanded instance.
#[derive(Debug, Clone)]
pub(crate) struct ExpandedInstance {
    pub component: Component,
    pub type_name: String,
    pub targetable: Vec<(Targetable, Origin)>,
    pub catalogue: Vec<(CatalogueMode, Origin)>,
    /// Element path -> origin, for every element of the component.
    pub provenance: Vec<(String, Origin)>,
}

/// The libraries an expansion may draw on, with the one the system binds.
pub(crate) struct LibrarySet<'a> {
    pub libraries: &'a [Library],
}

impl<'a> LibrarySet<'a> {
    fn get(&self, name: &str) -> Option<&'a Library> {
        self.libraries.iter().find(|l| l.name == name)
    }

    /// Resolves a fragment reference written inside `from`.
    fn fragment(&self, from: &'a Library, reference: &str) -> Result<(&'a Library, usize), String> {
        let (library, name) = match reference.split_once('.') {
            Some((lib, name)) => {
                if lib != from.name && !from.imports.iter().any(|i| i == lib) {
                    return Err(format!(
                        "fragment `{reference}` names library `{lib}`, which library `{}` does not import",
                        from.name
                    ));
                }
                let library = self
                    .get(lib)
                    .ok_or_else(|| format!("library `{lib}` is not available"))?;
                (library, name)
            }
            None => (from, reference),
        };
        let index = library
            .fragments
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| format!("library `{}` has no fragment `{name}`", library.name))?;
        Ok((library, index))
    }
}

/// A piece of body with the origin of each element.
#[derive(Default)]
struct Piece {
    parts: BodyParts,
    targetable: Vec<(Targetable, Origin)>,
    origins: Vec<(Category, String, Origin)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Category {
    Attribute,
    Port,
    Interface,
    Automaton,
    Function,
    Equation,
    Allocation,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::Attribute => "attribute",
            Category::Port => "port",
            Category::Interface => "interface",
            Category::Automaton => "automaton",
            Category::Function => "sensitive function",
            Category::Equation => "equation of",
            Category::Allocation => "allocation",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Category::Attribute => "attribute",
            Category::Port => "port",
            Category::Interface => "interface",
            Category::Automaton => "automaton",
            Category::Function => "function",
            Category::Equation => "equation",
            Category::Allocation => "allocation",
        }
    }
}

fn names_of(parts: &BodyParts) -> Vec<(Category, String)> {
    let mut out = Vec::new();
    out.extend(
        parts
            .attributes
            .iter()
            .map(|a| (Category::Attribute, a.name.clone())),
    );
    out.extend(parts.ports.iter().map(|p| (Category::Port, p.name.clone())));
    out.extend(
        parts
            .interfaces
            .iter()
            .map(|i| (Category::Interface, i.name.clone())),
    );
    out.extend(
        parts
            .automata
            .iter()
            .map(|a| (Category::Automaton, a.name.clone())),
    );
    out.extend(
        parts
            .sensitive_functions
            .iter()
            .map(|f| (Category::Function, f.name.clone())),
    );
    out.extend(
        parts
            .equations
            .iter()
            .map(|e| (Category::Equation, e.target.clone())),
    );
    out.extend(
        parts
            .allocations
            .iter()
            .map(|a| (Category::Allocation, a.name.clone())),
    );
    out
}

/// Merges `other` into `into`, refusing any name both declare.
fn merge(into: &mut Piece, other: Piece, ctx: &Ctx<'_>) -> Result<(), ExpandError> {
    for (category, name, origin) in &other.origins {
        if let Some((_, _, first)) = into
            .origins
            .iter()
            .find(|(c, n, _)| c == category && n == name)
        {
            return Err(ExpandError::new(
                origin.document,
                origin.pointer.clone(),
                format!(
                    "{}: {} `{name}` is declared by both {} and {}",
                    ctx.context,
                    category.label(),
                    first.describe(),
                    origin.describe()
                ),
            ));
        }
    }
    for (target, origin) in &other.targetable {
        if let Some((_, first)) = into
            .targetable
            .iter()
            .find(|(t, _)| t.attribute == target.attribute)
        {
            return Err(ExpandError::new(
                origin.document,
                origin.pointer.clone(),
                format!(
                    "{}: targetable attribute `{}` is declared by both {} and {}",
                    ctx.context,
                    target.attribute,
                    first.describe(),
                    origin.describe()
                ),
            ));
        }
    }
    let parts = other.parts;
    into.parts.attributes.extend(parts.attributes);
    into.parts.ports.extend(parts.ports);
    into.parts.interfaces.extend(parts.interfaces);
    into.parts.automata.extend(parts.automata);
    into.parts
        .sensitive_functions
        .extend(parts.sensitive_functions);
    into.parts.equations.extend(parts.equations);
    into.parts.allocations.extend(parts.allocations);
    into.targetable.extend(other.targetable);
    into.origins.extend(other.origins);
    Ok(())
}

/// Expands the body and targetable list of a type or fragment, already
/// bound to `env`, into a piece.
fn own_piece(
    body: &Value,
    targetable: &Value,
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
    origin: &Origin,
) -> Result<Piece, ExpandError> {
    let body_pointer = pointer_key(pointer, "body");
    let body = substitute(body, env, ctx, &body_pointer)?;
    let parts: BodyParts = typed(&body, DocumentKind::Library, &body_pointer, &ctx.context)?;
    let origins = names_of(&parts)
        .into_iter()
        .map(|(c, n)| (c, n, origin.at(body_pointer.clone())))
        .collect();
    let targetable_pointer = pointer_key(pointer, "targetable");
    let targetable = substitute(targetable, env, ctx, &targetable_pointer)?;
    let targetable: Vec<Targetable> = located(
        &targetable,
        DocumentKind::Library,
        &targetable_pointer,
        &ctx.context,
    )?;
    let targetable = targetable
        .into_iter()
        .map(|t| (t, origin.at(targetable_pointer.clone())))
        .collect();
    Ok(Piece {
        parts,
        targetable,
        origins,
    })
}

/// Expands the fragment applications `uses` of an owner bound to `env`.
#[allow(clippy::too_many_arguments)]
fn used_pieces(
    set: &LibrarySet<'_>,
    owner_library: &Library,
    uses: &[Use],
    env: &Env,
    ctx: &Ctx<'_>,
    pointer: &str,
    stack: &mut Vec<String>,
    origin: &Origin,
) -> Result<Vec<Piece>, ExpandError> {
    let mut pieces = Vec::new();
    for (j, application) in uses.iter().enumerate() {
        let use_pointer = pointer_index(&pointer_key(pointer, "uses"), j);
        let (library, index) = set
            .fragment(owner_library, &application.fragment)
            .map_err(|m| {
                ExpandError::new(
                    DocumentKind::Library,
                    format!("{use_pointer}/fragment"),
                    format!("{}: {m}", ctx.context),
                )
            })?;
        let fragment = &library.fragments[index];
        let qualified = format!("{}.{}", library.name, fragment.name);
        if stack.contains(&qualified) {
            let mut chain = stack.clone();
            chain.push(qualified);
            return Err(ExpandError::new(
                DocumentKind::Library,
                format!("{use_pointer}/fragment"),
                format!(
                    "{}: fragments apply each other in a cycle: {}",
                    ctx.context,
                    chain.join(" -> ")
                ),
            ));
        }
        let with_pointer = pointer_key(&use_pointer, "with");
        let mut given = Map::new();
        for (key, value) in &application.with {
            given.insert(
                key.clone(),
                substitute(value, env, ctx, &pointer_key(&with_pointer, key))?,
            );
        }
        let fragment_pointer = pointer_index("/library/fragments", index);
        let fragment_ctx = Ctx {
            document: ctx.document,
            instance: ctx.instance,
            context: format!("{} (fragment `{qualified}`)", ctx.context),
        };
        let fragment_env = bind_params(
            &fragment.parameters,
            &given,
            &with_pointer,
            &pointer_key(&fragment_pointer, "parameters"),
            &fragment_ctx,
        )?;
        let fragment_origin = origin.through_fragment(&qualified, &library.name);
        stack.push(qualified);
        let mut piece = own_piece(
            &fragment.body,
            &fragment.targetable,
            &fragment_env,
            &fragment_ctx,
            &fragment_pointer,
            &fragment_origin,
        )?;
        for inner in used_pieces(
            set,
            library,
            &fragment.uses,
            &fragment_env,
            &fragment_ctx,
            &fragment_pointer,
            stack,
            &fragment_origin,
        )? {
            merge(&mut piece, inner, &fragment_ctx)?;
        }
        stack.pop();
        pieces.push(piece);
    }
    Ok(pieces)
}

fn bind_params(
    params: &[Parameter],
    given: &Map<String, Value>,
    given_pointer: &str,
    params_pointer: &str,
    ctx: &Ctx<'_>,
) -> Result<Env, ExpandError> {
    bind(
        params,
        given,
        DocumentKind::Library,
        given_pointer,
        params_pointer,
        ctx,
    )
}

/// Every attribute, port and state reference an expression of `component`
/// makes must stay inside the component: a type reaches its neighbours
/// through ports only, reading them with `port_agg` and writing them with
/// an edge effect through a port (`raichu_model::PORT`).
fn check_self_references(
    component: &Component,
    ctx: &Ctx<'_>,
    pointer: &str,
) -> Result<(), ExpandError> {
    let me = component.name.as_str();
    let mut exprs: Vec<&Expr> = Vec::new();
    let mut targets: Vec<&raichu_expr::AttrRef> = Vec::new();
    for automaton in &component.automata {
        for transition in &automaton.transitions {
            exprs.extend(transition.guard.iter());
            if let raichu_model::Distrib::Exp {
                rate_expr: Some(e), ..
            } = &transition.distrib
            {
                exprs.push(e);
            }
            for effect in &transition.effects {
                exprs.push(&effect.value);
                targets.push(&effect.target);
            }
        }
    }
    for function in &component.sensitive_functions {
        for effect in &function.effects {
            exprs.push(&effect.value);
            targets.push(&effect.target);
        }
    }
    exprs.extend(component.equations.iter().map(|e| &e.expr));
    exprs.extend(component.allocations.iter().map(|a| &a.available));
    // Each expression with whether it may read `$edge`: only a port
    // equation reads the channels of its own connection.
    let checked = exprs.into_iter().map(|e| (e, false)).chain(
        component
            .ports
            .iter()
            .flat_map(|p| p.equations.iter().map(|e| (&e.expr, true))),
    );
    // An effect target naming `$port` writes through one of the
    // component's own in ports: the core resolves it to the connected
    // neighbours and refuses it outside an edge effect.
    let mut found: Vec<String> = targets
        .iter()
        .filter(|t| t.component != me && t.component != raichu_model::PORT)
        .map(|t| format!("`{}.{}`", t.component, t.attribute))
        .collect();
    for (expr, edge_allowed) in checked {
        expr.for_each_attr_ref(&mut |r| {
            if r.component != me && !(edge_allowed && r.component == raichu_model::EDGE) {
                found.push(format!("`{}.{}`", r.component, r.attribute));
            }
        });
        expr.for_each_port_ref(&mut |r| {
            if r.component != me {
                found.push(format!("port `{}.{}`", r.component, r.port));
            }
        });
        expr.for_each_state_ref(&mut |r| {
            if r.component != me {
                found.push(format!(
                    "state `{}.{}.{}`",
                    r.component, r.automaton, r.state
                ));
            }
        });
    }
    let foreign = found.into_iter().next();
    match foreign {
        Some(what) => Err(ExpandError::new(
            DocumentKind::Library,
            pointer,
            format!(
                "{}: the type body refers to {what} outside its own instance; write `$self` for the instance, and reach a neighbour through a port",
                ctx.context
            ),
        )),
        None => Ok(()),
    }
}

/// Expands one instance of `ty` (index `type_index` in `library`).
pub(crate) fn instantiate(
    set: &LibrarySet<'_>,
    library: &Library,
    type_index: usize,
    instance: &Instance,
    instance_pointer: &str,
) -> Result<ExpandedInstance, ExpandError> {
    let ty: &TypeDef = &library.types[type_index];
    let type_pointer = pointer_index("/library/types", type_index);
    let ctx = Ctx {
        document: DocumentKind::Library,
        instance: &instance.name,
        context: format!("instance `{}` of type `{}`", instance.name, ty.name),
    };
    let env = bind(
        &ty.parameters,
        &instance.parameters,
        DocumentKind::System,
        &pointer_key(instance_pointer, "parameters"),
        &pointer_key(&type_pointer, "parameters"),
        &ctx,
    )?;
    let origin = Origin::type_body(&library.name, &instance.name, &ty.name, &type_pointer);
    let mut stack = Vec::new();
    let mut piece = Piece::default();
    for used in used_pieces(
        set,
        library,
        &ty.uses,
        &env,
        &ctx,
        &type_pointer,
        &mut stack,
        &origin,
    )? {
        merge(&mut piece, used, &ctx)?;
    }
    let own = own_piece(&ty.body, &ty.targetable, &env, &ctx, &type_pointer, &origin)?;
    merge(&mut piece, own, &ctx)?;

    let modes_pointer = pointer_key(&type_pointer, "modes");
    let modes = substitute(&ty.modes, &env, &ctx, &modes_pointer)?;
    let modes: Vec<CatalogueMode> =
        located(&modes, DocumentKind::Library, &modes_pointer, &ctx.context)?;
    for (i, mode) in modes.iter().enumerate() {
        if modes[..i].iter().any(|m| m.name == mode.name) {
            return Err(ExpandError::new(
                DocumentKind::Library,
                format!("{}/name", pointer_index(&modes_pointer, i)),
                format!(
                    "{}: mode `{}` is declared twice on type `{}`",
                    ctx.context, mode.name, ty.name
                ),
            ));
        }
    }
    let catalogue = modes
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            let o = origin
                .at(pointer_index(&modes_pointer, i))
                .with_mode(&m.name);
            (m, o)
        })
        .collect();

    let parts = piece.parts;
    let component = Component {
        name: instance.name.clone(),
        attributes: parts.attributes,
        ports: parts.ports,
        interfaces: parts.interfaces,
        automata: parts.automata,
        sensitive_functions: parts.sensitive_functions,
        equations: parts.equations,
        allocations: parts.allocations,
    };
    check_self_references(&component, &ctx, &pointer_key(&type_pointer, "body"))?;
    let mut provenance = vec![(
        format!("component:{}", instance.name),
        origin.at(instance_pointer.to_owned()).in_system(),
    )];
    for (category, name, o) in piece.origins {
        provenance.push((format!("{}:{}.{name}", category.key(), instance.name), o));
    }
    let mut transitions = Vec::new();
    for automaton in &component.automata {
        let key = format!("automaton:{}.{}", instance.name, automaton.name);
        if let Some(o) = provenance
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, o)| o.clone())
        {
            for transition in &automaton.transitions {
                transitions.push((
                    format!(
                        "transition:{}.{}.{}",
                        instance.name, automaton.name, transition.name
                    ),
                    o.clone(),
                ));
            }
        }
    }
    provenance.extend(transitions);
    Ok(ExpandedInstance {
        component,
        type_name: ty.name.clone(),
        targetable: piece.targetable,
        catalogue,
        provenance,
    })
}
