//! The OpenPSA model-exchange format, read and written.
//!
//! What is read: `define-fault-tree`, `define-gate`, `define-basic-event`,
//! `define-house-event` and `define-parameter`, wherever they sit under
//! `opsa-mef`; the connectives `and`, `or`, `not`, `atleast`, `nand`,
//! `nor`, `xor`, `iff`, `imply` and `cardinality`, rewritten onto
//! conjunction, disjunction, vote and negation; references by
//! `basic-event`, `gate`, `house-event` and `event`; `constant`.
//!
//! A basic event's expression becomes a [`Law`]: a number (`float`, `int`,
//! a parameter, or arithmetic over them) is a constant probability;
//! `exponential(λ, system-mission-time)` and
//! `Weibull(α, β, 0, system-mission-time)` keep their law, so the tree is
//! quantified at whatever mission time is asked; with a number in place of
//! the mission time they are evaluated once. An event carrying no
//! expression but the `law` attributes the engine's own writer emits for a
//! law OpenPSA has no expression for (log-normal, gamma, uniform, delay,
//! empirical) gets that law back. Anything else is refused by name rather
//! than read approximately.

use std::collections::HashMap;

use roxmltree::{Document, Node};

use crate::{BasicEvent, Formula, FtaError, GateDef, Law, Tree};

/// Read a fault tree. `top` names the gate to quantify; without it the
/// tree must have exactly one gate no other gate references.
pub fn read_open_psa(text: &str, top: Option<&str>) -> Result<Tree, FtaError> {
    // The formulas are read recursively, one frame per nesting level.
    crate::with_large_stack(|| read_here(text, top))
}

fn read_here(text: &str, top: Option<&str>) -> Result<Tree, FtaError> {
    let doc = Document::parse(text).map_err(|e| FtaError::OpenPsa(e.to_string()))?;
    let root = doc.root_element();
    if root.tag_name().name() != "opsa-mef" {
        return Err(FtaError::OpenPsa(format!(
            "the root element is <{}>, not <opsa-mef>",
            root.tag_name().name()
        )));
    }
    let mut reader = Reader::default();
    let mut tree_name = None;
    for node in root.descendants().filter(Node::is_element) {
        let name = node.attribute("name").unwrap_or_default();
        match node.tag_name().name() {
            // Both change what a basic event means; ignoring them would
            // quantify a different tree without a word.
            tag @ ("define-substitution" | "define-CCF-group") => {
                return Err(FtaError::OpenPsa(format!(
                    "<{tag}> (`{name}`) is not read: quantifying without it \
                     would change the result"
                )));
            }
            "define-fault-tree" if tree_name.is_none() => tree_name = Some(name.to_owned()),
            "define-basic-event" => {
                reader
                    .event_index
                    .insert(name.to_owned(), reader.event_nodes.len());
                reader.event_nodes.push(node);
            }
            "define-gate" => {
                reader
                    .gate_index
                    .insert(name.to_owned(), reader.gate_nodes.len());
                reader.gate_nodes.push(node);
            }
            "define-house-event" => {
                let value = match expression_child(node)
                    .filter(|c| c.tag_name().name() == "constant")
                    .and_then(|c| c.attribute("value"))
                {
                    Some("true") => true,
                    Some("false") => false,
                    _ => {
                        return Err(FtaError::OpenPsa(format!(
                            "house event `{name}` needs <constant value=\"true\"/> \
                             or <constant value=\"false\"/>"
                        )))
                    }
                };
                if reader.house.insert(name.to_owned(), value).is_some() {
                    return Err(FtaError::OpenPsa(format!(
                        "house event `{name}` is defined twice"
                    )));
                }
            }
            "define-parameter" => {
                if reader.parameters.insert(name.to_owned(), node).is_some() {
                    return Err(FtaError::OpenPsa(format!(
                        "parameter `{name}` is defined twice"
                    )));
                }
            }
            _ => {}
        }
    }

    let mut events = Vec::with_capacity(reader.event_nodes.len());
    for node in &reader.event_nodes {
        let name = node.attribute("name").unwrap_or_default().to_owned();
        let law = reader.law(*node, &name)?;
        events.push(BasicEvent { name, law });
    }
    let mut gates = Vec::with_capacity(reader.gate_nodes.len());
    let mut referenced = vec![false; reader.gate_nodes.len()];
    for node in &reader.gate_nodes {
        let name = node.attribute("name").unwrap_or_default().to_owned();
        let body = expression_child(*node)
            .ok_or_else(|| FtaError::OpenPsa(format!("gate `{name}` has no formula")))?;
        let formula = reader.formula(body, &mut referenced)?;
        gates.push(GateDef { name, formula });
    }
    let top = match top {
        Some(name) => Formula::Gate {
            gate: *reader
                .gate_index
                .get(name)
                .ok_or_else(|| FtaError::OpenPsa(format!("no gate is named `{name}`")))?,
        },
        None => {
            let roots: Vec<usize> = (0..gates.len()).filter(|&g| !referenced[g]).collect();
            match roots.as_slice() {
                [gate] => Formula::Gate { gate: *gate },
                [] => return Err(FtaError::OpenPsa("the document defines no gate".to_owned())),
                many => {
                    let names: Vec<&str> = many.iter().map(|&g| gates[g].name.as_str()).collect();
                    return Err(FtaError::OpenPsa(format!(
                        "several gates are referenced by no other ({}); name the \
                         top gate",
                        names.join(", ")
                    )));
                }
            }
        }
    };
    let tree = Tree {
        name: tree_name.unwrap_or_else(|| "fault_tree".to_owned()),
        events,
        gates,
        top,
    };
    tree.validate()?;
    Ok(tree)
}

#[derive(Default)]
struct Reader<'a, 'input> {
    event_nodes: Vec<Node<'a, 'input>>,
    event_index: HashMap<String, usize>,
    gate_nodes: Vec<Node<'a, 'input>>,
    gate_index: HashMap<String, usize>,
    house: HashMap<String, bool>,
    parameters: HashMap<String, Node<'a, 'input>>,
    /// The parameters being evaluated, outermost first: a parameter met
    /// again on this path is a cycle.
    evaluating: std::cell::RefCell<Vec<String>>,
}

/// The first element child that is neither a label nor attributes.
fn expression_child<'a, 'input>(node: Node<'a, 'input>) -> Option<Node<'a, 'input>> {
    node.children()
        .filter(Node::is_element)
        .find(|c| !matches!(c.tag_name().name(), "label" | "attributes"))
}

fn arguments<'a, 'input>(node: Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
    node.children()
        .filter(Node::is_element)
        .filter(|c| !matches!(c.tag_name().name(), "label" | "attributes"))
        .collect()
}

fn xor(a: Formula, b: Formula) -> Formula {
    Formula::or(vec![
        Formula::and(vec![a.clone(), Formula::not(b.clone())]),
        Formula::and(vec![Formula::not(a), b]),
    ])
}

impl Reader<'_, '_> {
    fn formula(&self, node: Node, referenced: &mut [bool]) -> Result<Formula, FtaError> {
        let tag = node.tag_name().name();
        let name = node.attribute("name").unwrap_or_default();
        let walk = |referenced: &mut [bool]| -> Result<Vec<Formula>, FtaError> {
            arguments(node)
                .into_iter()
                .map(|a| self.formula(a, referenced))
                .collect()
        };
        let two = |mut v: Vec<Formula>, what: &str| -> Result<(Formula, Formula), FtaError> {
            if v.len() != 2 {
                return Err(FtaError::OpenPsa(format!(
                    "<{what}> takes two arguments, got {}",
                    v.len()
                )));
            }
            let b = v.pop().unwrap_or(Formula::Constant { value: false });
            let a = v.pop().unwrap_or(Formula::Constant { value: false });
            Ok((a, b))
        };
        let count = |attr: &str| -> Result<usize, FtaError> {
            node.attribute(attr)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| FtaError::OpenPsa(format!("<{tag}> needs an integer `{attr}`")))
        };
        Ok(match tag {
            "and" => Formula::and(walk(referenced)?),
            "or" => Formula::or(walk(referenced)?),
            "nand" => Formula::not(Formula::and(walk(referenced)?)),
            "nor" => Formula::not(Formula::or(walk(referenced)?)),
            "not" => {
                let mut v = walk(referenced)?;
                if v.len() != 1 {
                    return Err(FtaError::OpenPsa(format!(
                        "<not> takes one argument, got {}",
                        v.len()
                    )));
                }
                Formula::not(v.pop().unwrap_or(Formula::Constant { value: false }))
            }
            "atleast" => Formula::at_least(count("min")?, walk(referenced)?),
            "cardinality" => {
                let (min, max) = (count("min")?, count("max")?);
                let v = walk(referenced)?;
                Formula::and(vec![
                    Formula::at_least(min, v.clone()),
                    Formula::not(Formula::at_least(max + 1, v)),
                ])
            }
            // Binary only: an n-ary xor is read as parity by some tools and
            // as "exactly one" by others, and its rewriting doubles at each
            // argument.
            "xor" => {
                let (a, b) = two(walk(referenced)?, "xor")?;
                xor(a, b)
            }
            "iff" => {
                let (a, b) = two(walk(referenced)?, "iff")?;
                Formula::not(xor(a, b))
            }
            "imply" => {
                let (a, b) = two(walk(referenced)?, "imply")?;
                Formula::or(vec![Formula::not(a), b])
            }
            "constant" => Formula::Constant {
                value: node.attribute("value") == Some("true"),
            },
            "basic-event" => self.event_ref(name)?,
            "gate" => self.gate_ref(name, referenced)?,
            "house-event" => self.house_ref(name)?,
            "event" => {
                let kinds = [
                    self.gate_index.contains_key(name),
                    self.house.contains_key(name),
                    self.event_index.contains_key(name),
                ];
                match kinds {
                    [true, false, false] => self.gate_ref(name, referenced)?,
                    [false, true, false] => self.house_ref(name)?,
                    [false, false, _] => self.event_ref(name)?,
                    _ => {
                        return Err(FtaError::OpenPsa(format!(
                            "<event name=\"{name}\"> is ambiguous: several \
                             definitions carry that name"
                        )))
                    }
                }
            }
            other => {
                return Err(FtaError::OpenPsa(format!(
                    "the connective <{other}> is not read"
                )))
            }
        })
    }

    fn event_ref(&self, name: &str) -> Result<Formula, FtaError> {
        self.event_index
            .get(name)
            .map(|&event| Formula::Event { event })
            .ok_or_else(|| FtaError::OpenPsa(format!("undefined basic event `{name}`")))
    }

    fn gate_ref(&self, name: &str, referenced: &mut [bool]) -> Result<Formula, FtaError> {
        let gate = *self
            .gate_index
            .get(name)
            .ok_or_else(|| FtaError::OpenPsa(format!("undefined gate `{name}`")))?;
        referenced[gate] = true;
        Ok(Formula::Gate { gate })
    }

    fn house_ref(&self, name: &str) -> Result<Formula, FtaError> {
        self.house
            .get(name)
            .map(|&value| Formula::Constant { value })
            .ok_or_else(|| FtaError::OpenPsa(format!("undefined house event `{name}`")))
    }

    fn law(&self, node: Node, event: &str) -> Result<Law, FtaError> {
        let unsupported =
            |detail: String| FtaError::OpenPsa(format!("basic event `{event}`: {detail}"));
        let Some(expr) = expression_child(node) else {
            return self.attributes_law(node, event);
        };
        match expr.tag_name().name() {
            "exponential" => {
                let args = arguments(expr);
                if args.len() != 2 {
                    return Err(unsupported("<exponential> takes two arguments".to_owned()));
                }
                let rate = self.number(args[0], event)?;
                if args[1].tag_name().name() == "system-mission-time" {
                    Ok(Law::Exponential { rate })
                } else {
                    let t = self.number(args[1], event)?;
                    Ok(Law::Constant {
                        probability: -libm::expm1(-rate * t),
                    })
                }
            }
            "Weibull" => {
                let args = arguments(expr);
                if args.len() != 4 {
                    return Err(unsupported("<Weibull> takes four arguments".to_owned()));
                }
                let scale = self.number(args[0], event)?;
                let shape = self.number(args[1], event)?;
                let shift = self.number(args[2], event)?;
                if shift != 0.0 {
                    return Err(unsupported(format!(
                        "a Weibull law shifted by {shift} is not read"
                    )));
                }
                let law = Law::Weibull { shape, scale };
                if args[3].tag_name().name() == "system-mission-time" {
                    Ok(law)
                } else {
                    let t = self.number(args[3], event)?;
                    Ok(Law::Constant {
                        probability: law.probability(event, Some(t))?,
                    })
                }
            }
            _ => Ok(Law::Constant {
                probability: self.number(expr, event)?,
            }),
        }
    }

    fn number(&self, node: Node, event: &str) -> Result<f64, FtaError> {
        let bad = |detail: String| FtaError::OpenPsa(format!("basic event `{event}`: {detail}"));
        let tag = node.tag_name().name();
        let values = || -> Result<Vec<f64>, FtaError> {
            arguments(node)
                .into_iter()
                .map(|a| self.number(a, event))
                .collect()
        };
        match tag {
            "float" | "int" => node
                .attribute("value")
                .and_then(|v| v.parse::<f64>().ok())
                .ok_or_else(|| bad(format!("<{tag}> without a numeric value"))),
            "parameter" => {
                let name = node.attribute("name").unwrap_or_default();
                let definition = self
                    .parameters
                    .get(name)
                    .ok_or_else(|| bad(format!("undefined parameter `{name}`")))?;
                let expr = expression_child(*definition)
                    .ok_or_else(|| bad(format!("parameter `{name}` has no value")))?;
                if self.evaluating.borrow().iter().any(|p| p == name) {
                    let mut path = self.evaluating.borrow().clone();
                    path.push(name.to_owned());
                    return Err(bad(format!("parameter cycle {}", path.join(" -> "))));
                }
                self.evaluating.borrow_mut().push(name.to_owned());
                let value = self.number(expr, event);
                self.evaluating.borrow_mut().pop();
                value
            }
            "mul" => Ok(values()?.iter().product()),
            "add" => Ok(values()?.iter().sum()),
            "sub" => {
                let v = values()?;
                let (first, rest) = v
                    .split_first()
                    .ok_or_else(|| bad("<sub> without arguments".to_owned()))?;
                Ok(rest.iter().fold(*first, |acc, x| acc - x))
            }
            "div" => {
                let v = values()?;
                let (first, rest) = v
                    .split_first()
                    .ok_or_else(|| bad("<div> without arguments".to_owned()))?;
                Ok(rest.iter().fold(*first, |acc, x| acc / x))
            }
            "neg" => Ok(-values()?.first().copied().unwrap_or(0.0)),
            "exponential" | "Weibull" => Err(bad(format!(
                "<{tag}> is only read as the whole expression of a basic event"
            ))),
            other => Err(bad(format!("the expression <{other}> is not read"))),
        }
    }

    fn attributes_law(&self, node: Node, event: &str) -> Result<Law, FtaError> {
        let missing = || {
            FtaError::OpenPsa(format!(
                "basic event `{event}` carries neither an expression nor a law"
            ))
        };
        let attributes = node
            .children()
            .find(|c| c.tag_name().name() == "attributes")
            .ok_or_else(missing)?;
        let mut map = serde_json::Map::new();
        for attr in attributes
            .children()
            .filter(|c| c.tag_name().name() == "attribute")
        {
            let (Some(key), Some(value)) = (attr.attribute("name"), attr.attribute("value")) else {
                continue;
            };
            let value = serde_json::from_str(value)
                .unwrap_or_else(|_| serde_json::Value::String(value.to_owned()));
            map.insert(key.to_owned(), value);
        }
        if !map.contains_key("law") {
            return Err(missing());
        }
        law_from_json(serde_json::Value::Object(map))
            .map_err(|detail| FtaError::OpenPsa(format!("basic event `{event}`: {detail}")))
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "law", rename_all = "snake_case")]
enum JsonLaw {
    Lognormal { mu: f64, sigma: f64 },
    Gamma { shape: f64, scale: f64 },
    Uniform { low: f64, high: f64 },
    Delay { time: f64 },
    Empirical { points: Vec<(f64, f64)> },
    Exponential { rate: f64 },
    Weibull { shape: f64, scale: f64 },
}

fn law_from_json(value: serde_json::Value) -> Result<Law, String> {
    let law: JsonLaw = serde_json::from_value(value).map_err(|e| e.to_string())?;
    Ok(match law {
        JsonLaw::Lognormal { mu, sigma } => Law::Lognormal { mu, sigma },
        JsonLaw::Gamma { shape, scale } => Law::Gamma { shape, scale },
        JsonLaw::Uniform { low, high } => Law::Uniform { low, high },
        JsonLaw::Delay { time } => Law::Delay { time },
        JsonLaw::Empirical { points } => Law::Empirical { points },
        JsonLaw::Exponential { rate } => Law::Exponential { rate },
        JsonLaw::Weibull { shape, scale } => Law::Weibull { shape, scale },
    })
}

/// Write `tree` in the OpenPSA format, its top as a gate named `top` (or
/// `top_` and so on when a gate already carries that name). A law OpenPSA
/// has no expression for is carried as `law` attributes, which
/// [`read_open_psa`] reads back.
pub fn write_open_psa(tree: &Tree) -> String {
    let mut top_name = "top".to_owned();
    while tree.gates.iter().any(|g| g.name == top_name) {
        top_name.push('_');
    }
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<opsa-mef>\n");
    out.push_str(&format!(
        "  <define-fault-tree name=\"{}\">\n",
        xml(&tree.name)
    ));
    out.push_str(&format!(
        "    <define-gate name=\"{}\">\n      {}\n    </define-gate>\n",
        xml(&top_name),
        formula_xml(&tree.top, tree)
    ));
    for gate in &tree.gates {
        out.push_str(&format!(
            "    <define-gate name=\"{}\">\n      {}\n    </define-gate>\n",
            xml(&gate.name),
            formula_xml(&gate.formula, tree)
        ));
    }
    out.push_str("  </define-fault-tree>\n  <model-data>\n");
    for event in &tree.events {
        out.push_str(&format!(
            "    <define-basic-event name=\"{}\">\n{}    </define-basic-event>\n",
            xml(&event.name),
            law_xml(&event.law)
        ));
    }
    out.push_str("  </model-data>\n</opsa-mef>\n");
    out
}

fn xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn formula_xml(formula: &Formula, tree: &Tree) -> String {
    let join = |args: &[Formula]| -> String {
        args.iter()
            .map(|a| formula_xml(a, tree))
            .collect::<Vec<_>>()
            .join("")
    };
    match formula {
        Formula::Constant { value } => format!("<constant value=\"{value}\"/>"),
        Formula::Event { event } => {
            format!("<basic-event name=\"{}\"/>", xml(&tree.events[*event].name))
        }
        Formula::Gate { gate } => format!("<gate name=\"{}\"/>", xml(&tree.gates[*gate].name)),
        Formula::And { args } => format!("<and>{}</and>", join(args)),
        Formula::Or { args } => format!("<or>{}</or>", join(args)),
        Formula::AtLeast { k, args } => format!("<atleast min=\"{k}\">{}</atleast>", join(args)),
        Formula::Not { arg } => format!("<not>{}</not>", formula_xml(arg, tree)),
    }
}

fn law_xml(law: &Law) -> String {
    let float = |v: f64| format!("<float value=\"{v}\"/>");
    match law {
        Law::Constant { probability } => format!("      {}\n", float(*probability)),
        Law::Exponential { rate } => format!(
            "      <exponential>{}<system-mission-time/></exponential>\n",
            float(*rate)
        ),
        Law::Weibull { shape, scale } => format!(
            "      <Weibull>{}{}{}<system-mission-time/></Weibull>\n",
            float(*scale),
            float(*shape),
            float(0.0)
        ),
        other => {
            let json = serde_json::to_value(other).unwrap_or_default();
            let mut attributes = String::from("      <attributes>\n");
            if let Some(map) = json.as_object() {
                for (key, value) in map {
                    let text = match value {
                        serde_json::Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    attributes.push_str(&format!(
                        "        <attribute name=\"{}\" value=\"{}\"/>\n",
                        xml(key),
                        xml(&text)
                    ));
                }
            }
            attributes.push_str("      </attributes>\n");
            attributes
        }
    }
}
