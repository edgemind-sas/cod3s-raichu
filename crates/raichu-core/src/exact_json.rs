//! A JSON reader whose floating-point numbers are correctly rounded.
//!
//! `serde_json` parses a float to within one unit in the last place
//! unless its `float_roundtrip` feature is on, and a feature is unified
//! across everything built together: switching it on for one crate would
//! switch it on for every model document the engine reads, which may move
//! a parsed parameter by one ulp and a result with it. The readers of the
//! engine's own result documents (`raichu.exploration`,
//! `raichu.quantification`, `raichu.fault_tree`, the `raichu.sequences`
//! header) parse through this module instead: numbers go through
//! [`str::parse`], which is correctly rounded, so a written document reads
//! back to the same bits, and nothing else in the workspace is touched.
//!
//! [`from_str`] deserializes straight into a typed result and keeps
//! integers past `u64`; every reader above goes through it, so they share
//! one integer policy. [`parse`] returns a [`serde_json::Value`], which
//! cannot hold an integer past `u64`: it serves code that needs the
//! document itself (the canonical form a model hash is computed on).

use serde::de::value::{MapAccessDeserializer, MapDeserializer, SeqDeserializer};
use serde::de::{DeserializeOwned, IntoDeserializer, Visitor};
use serde_json::{Map, Number, Value};

/// Why a document could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message} at byte {offset}")]
pub struct ParseError {
    /// What is wrong.
    pub message: String,
    /// Byte offset in the text where it was found.
    pub offset: usize,
}

/// Why a document could not be read into a typed result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FromStrError {
    /// The text is not JSON this module accepts.
    #[error(transparent)]
    Parse(#[from] ParseError),
    /// The JSON does not have the shape of the requested type.
    #[error("{0}")]
    Shape(String),
}

/// Parse `text` as one JSON value, numbers correctly rounded. Accepts the
/// documents `serde_json` accepts, except that a member name repeated in
/// one object is refused (`serde_json` keeps the last one silently, which
/// the typed readers built on this module would otherwise inherit), and
/// nesting is capped at 256 levels. An integer past the `i64`/`u64` range
/// becomes the nearest float, as `serde_json` reads it.
///
/// # Errors
/// [`ParseError`], naming what is wrong and where.
pub fn parse(text: &str) -> Result<Value, ParseError> {
    Ok(document(text)?.into_value())
}

/// Read `text` into a `T`, numbers correctly rounded. Accepts what
/// [`parse`] accepts, and reads an integer past `u64` the same way (the
/// nearest float) unless the target field is a `u128` or an `i128`, which
/// gets it exactly.
///
/// # Errors
/// [`FromStrError::Parse`] when the text is not accepted,
/// [`FromStrError::Shape`] when it does not have the shape of `T`.
pub fn from_str<T: DeserializeOwned>(text: &str) -> Result<T, FromStrError> {
    T::deserialize(document(text)?).map_err(|e| FromStrError::Shape(e.0))
}

fn document(text: &str) -> Result<Node, ParseError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        text,
        at: 0,
    };
    let node = parser.value(0)?;
    parser.skip_whitespace();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(node)
}

/// One parsed JSON value, integers kept to 128 bits.
#[derive(Debug)]
enum Node {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    U128(u128),
    I128(i128),
    F64(f64),
    String(String),
    Array(Vec<Node>),
    Object(Vec<(String, Node)>),
}

impl Node {
    fn into_value(self) -> Value {
        match self {
            Node::Null => Value::Null,
            Node::Bool(b) => Value::Bool(b),
            Node::U64(n) => Value::Number(n.into()),
            Node::I64(n) => Value::Number(n.into()),
            // Rounded to nearest, as the float parse of the token would be.
            Node::U128(n) => float(n as f64),
            Node::I128(n) => float(n as f64),
            Node::F64(x) => float(x),
            Node::String(s) => Value::String(s),
            Node::Array(items) => Value::Array(items.into_iter().map(Node::into_value).collect()),
            Node::Object(members) => Value::Object(
                members
                    .into_iter()
                    .map(|(k, v)| (k, v.into_value()))
                    .collect::<Map<_, _>>(),
            ),
        }
    }
}

/// A finite float as a JSON number (the parser refuses non-finite ones).
fn float(x: f64) -> Value {
    Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// A shape error raised while deserializing a [`Node`].
#[derive(Debug)]
struct ShapeError(String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

impl serde::de::Error for ShapeError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        ShapeError(msg.to_string())
    }
}

impl<'de> IntoDeserializer<'de, ShapeError> for Node {
    type Deserializer = Node;

    fn into_deserializer(self) -> Node {
        self
    }
}

impl<'de> serde::Deserializer<'de> for Node {
    type Error = ShapeError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ShapeError> {
        match self {
            Node::Null => visitor.visit_unit(),
            Node::Bool(b) => visitor.visit_bool(b),
            Node::U64(n) => visitor.visit_u64(n),
            Node::I64(n) => visitor.visit_i64(n),
            // Past `u64`, a target that does not ask for 128 bits gets the
            // nearest float, as from `serde_json` (whose `Value` cannot
            // hold more); `deserialize_u128`/`deserialize_i128` keep them.
            Node::U128(n) => visitor.visit_f64(n as f64),
            Node::I128(n) => visitor.visit_f64(n as f64),
            Node::F64(x) => visitor.visit_f64(x),
            Node::String(s) => visitor.visit_string(s),
            Node::Array(items) => {
                let mut seq = SeqDeserializer::new(items.into_iter());
                let value = visitor.visit_seq(&mut seq)?;
                seq.end()?;
                Ok(value)
            }
            Node::Object(members) => {
                let mut map = MapDeserializer::new(members.into_iter());
                let value = visitor.visit_map(&mut map)?;
                map.end()?;
                Ok(value)
            }
        }
    }

    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ShapeError> {
        match self {
            Node::U128(n) => visitor.visit_u128(n),
            node => node.deserialize_any(visitor),
        }
    }

    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ShapeError> {
        match self {
            Node::I128(n) => visitor.visit_i128(n),
            node => node.deserialize_any(visitor),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ShapeError> {
        match self {
            Node::Null => visitor.visit_none(),
            node => visitor.visit_some(node),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, ShapeError> {
        visitor.visit_newtype_struct(self)
    }

    /// An externally tagged enum: a unit variant as its name, any other
    /// variant as a one-member object.
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ShapeError> {
        match self {
            Node::String(s) => visitor.visit_enum(s.into_deserializer()),
            Node::Object(members) if members.len() == 1 => visitor.visit_enum(
                MapAccessDeserializer::new(MapDeserializer::new(members.into_iter())),
            ),
            _ => Err(serde::de::Error::custom(
                "expected an enum variant name or a one-member object",
            )),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct
        identifier ignored_any
    }
}

/// Nesting beyond this depth is refused rather than recursed into.
const MAX_DEPTH: usize = 256;

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> ParseError {
        ParseError {
            message: what.to_owned(),
            offset: self.at,
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.bytes.get(self.at) {
            self.at += 1;
        }
    }

    fn expect(&mut self, literal: &str, node: Node) -> Result<Node, ParseError> {
        if self.text[self.at..].starts_with(literal) {
            self.at += literal.len();
            Ok(node)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Node, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.skip_whitespace();
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Node::String),
            Some(b't') => self.expect("true", Node::Bool(true)),
            Some(b'f') => self.expect("false", Node::Bool(false)),
            Some(b'n') => self.expect("null", Node::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
            None => Err(self.error("unexpected end of document")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Node, ParseError> {
        self.at += 1;
        let mut members: Vec<(String, Node)> = Vec::new();
        let mut names = std::collections::HashSet::new();
        self.skip_whitespace();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Node::Object(members));
        }
        loop {
            self.skip_whitespace();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected a member name"));
            }
            let key_at = self.at;
            let key = self.string()?;
            if !names.insert(key.clone()) {
                return Err(ParseError {
                    message: format!("duplicate member `{key}`"),
                    offset: key_at,
                });
            }
            self.skip_whitespace();
            if self.bytes.get(self.at) != Some(&b':') {
                return Err(self.error("expected `:`"));
            }
            self.at += 1;
            let value = self.value(depth + 1)?;
            members.push((key, value));
            self.skip_whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Node::Object(members));
                }
                _ => return Err(self.error("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Node, ParseError> {
        self.at += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Node::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Node::Array(items));
                }
                _ => return Err(self.error("expected `,` or `]`")),
            }
        }
    }

    /// A string token, unescaped by `serde_json` (strings carry no
    /// rounding question).
    fn string(&mut self) -> Result<String, ParseError> {
        let start = self.at;
        self.at += 1;
        loop {
            match self.bytes.get(self.at) {
                Some(b'"') => {
                    self.at += 1;
                    break;
                }
                Some(b'\\') => self.at += 2,
                Some(_) => self.at += 1,
                None => return Err(self.error("unterminated string")),
            }
        }
        serde_json::from_str(&self.text[start..self.at]).map_err(|e| ParseError {
            message: e.to_string(),
            offset: start,
        })
    }

    fn number(&mut self) -> Result<Node, ParseError> {
        let start = self.at;
        while let Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') = self.bytes.get(self.at) {
            self.at += 1;
        }
        let token = &self.text[start..self.at];
        // Validate the token as JSON first (Rust's parsers also accept
        // `01`, `+1` or `1.`, which JSON does not).
        let invalid = |e: &dyn std::fmt::Display| ParseError {
            message: format!("{e} in number `{token}`"),
            offset: start,
        };
        serde_json::from_str::<Number>(token).map_err(|e| invalid(&e))?;
        if !token.contains(['.', 'e', 'E']) {
            if let Ok(n) = token.parse::<u64>() {
                return Ok(Node::U64(n));
            }
            if let Ok(n) = token.parse::<i64>() {
                return Ok(Node::I64(n));
            }
            if let Ok(n) = token.parse::<u128>() {
                return Ok(Node::U128(n));
            }
            if let Ok(n) = token.parse::<i128>() {
                return Ok(Node::I128(n));
            }
        }
        let x: f64 = token.parse().map_err(|e| invalid(&e))?;
        if x.is_finite() {
            Ok(Node::F64(x))
        } else {
            Err(ParseError {
                message: format!("number `{token}` is not finite"),
                offset: start,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{from_str, parse, FromStrError};
    use serde::{Deserialize, Serialize};

    /// The case `serde_json` without `float_roundtrip` reads one ulp off.
    #[test]
    fn floats_read_back_to_the_bits_they_were_written_from() {
        let x = 0.248_720_059_264_354_14_f64;
        let lossy: f64 = serde_json::from_str(&serde_json::to_string(&x).unwrap()).unwrap();
        assert_ne!(
            lossy.to_bits(),
            x.to_bits(),
            "not a case serde_json misreads"
        );
        let text = serde_json::to_string(&x).unwrap();
        let value = parse(&text).unwrap();
        assert_eq!(value.as_f64().unwrap().to_bits(), x.to_bits());
    }

    #[test]
    fn documents_parse_as_serde_json_parses_them() {
        let text = r#" {"a": [1, -2, 3.5e-3, true, null, "x\"é"], "b": {}, "c": []} "#;
        assert_eq!(
            parse(text).unwrap(),
            serde_json::from_str::<serde_json::Value>(text).unwrap()
        );
        for bad in [
            "{",
            "[1,]",
            "01",
            "01x",
            "tru",
            "1 2",
            "\"open",
            "+1",
            "{\"a\" 1}",
            "{\"a\": 1, \"a\": 2}",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        Unit,
        Pair(u32, String),
        Named { x: f64 },
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Tagged {
        Basic { name: String },
        Gate { inputs: Vec<String> },
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Typed {
        x: f64,
        count: Option<u128>,
        delta: i128,
        missing: Option<f64>,
        kinds: Vec<Kind>,
        tagged: Tagged,
        raw: serde_json::Value,
        #[serde(default)]
        defaulted: Vec<u8>,
    }

    /// A typed read keeps what a `Value` cannot: correctly rounded floats
    /// and integers past `u64`, through options, enums of every
    /// representation, nested values and defaults.
    #[test]
    fn typed_reads_round_trip_to_the_bits() {
        let typed = Typed {
            x: 0.248_720_059_264_354_14,
            count: Some(u128::from(u64::MAX) + 1),
            delta: i128::from(i64::MIN) - 1,
            missing: None,
            kinds: vec![
                Kind::Unit,
                Kind::Pair(3, "p".into()),
                Kind::Named { x: 0.1 },
            ],
            tagged: Tagged::Gate {
                inputs: vec!["a".into()],
            },
            raw: serde_json::json!({"order": [1, 2], "note": null}),
            defaulted: vec![],
        };
        let mut text = serde_json::to_string(&typed).unwrap();
        let back: Typed = from_str(&text).unwrap();
        assert_eq!(back.x.to_bits(), typed.x.to_bits());
        assert_eq!(back, typed);
        assert_eq!(serde_json::to_string(&back).unwrap(), text);
        // A member the type does not know is ignored, as serde_json does.
        text.insert_str(1, r#""extra":[1,{"y":2}],"#);
        assert_eq!(from_str::<Typed>(&text).unwrap(), typed);
    }

    #[test]
    fn typed_reads_name_a_wrong_shape_and_a_repeated_member() {
        assert!(matches!(
            from_str::<Typed>(r#"{"x": "one"}"#),
            Err(FromStrError::Shape(_))
        ));
        assert!(matches!(
            from_str::<Kind>(r#"{"unit": null, "pair": [1, "a"]}"#),
            Err(FromStrError::Shape(_))
        ));
        assert!(matches!(
            from_str::<serde_json::Value>(r#"{"a": 1, "a": 2}"#),
            Err(FromStrError::Parse(_))
        ));
    }

    /// An integer past `u64` reads as `serde_json` reads it (the nearest
    /// float) wherever the target does not ask for 128 bits, so the typed
    /// and the `Value` routes agree; a target that asks for 128 bits gets
    /// them, and one that cannot hold the value is refused by shape.
    #[test]
    fn integers_past_u64_read_as_serde_json_reads_them() {
        let big = "18446744073709551616";
        let expected = serde_json::from_str::<serde_json::Value>(big).unwrap();
        assert_eq!(parse(big).unwrap(), expected);
        assert_eq!(from_str::<serde_json::Value>(big).unwrap(), expected);
        assert_eq!(from_str::<f64>(big).unwrap(), 18_446_744_073_709_551_616.0);
        assert_eq!(from_str::<u128>(big).unwrap(), u128::from(u64::MAX) + 1);
        assert_eq!(
            from_str::<Option<i128>>("-9223372036854775809").unwrap(),
            Some(i128::from(i64::MIN) - 1)
        );
        assert!(matches!(from_str::<u64>(big), Err(FromStrError::Shape(_))));
        // Past `u128`, the token is a float, which no integer field takes.
        let huge = "1".repeat(40);
        assert!(matches!(
            from_str::<u128>(&huge),
            Err(FromStrError::Shape(_))
        ));
    }

    #[test]
    fn typed_reads_refuse_what_parse_refuses() {
        let deep = format!("{}{}", "[".repeat(300), "]".repeat(300));
        assert!(matches!(
            from_str::<serde_json::Value>(&deep),
            Err(FromStrError::Parse(_))
        ));
        assert!(matches!(
            from_str::<f64>("1e400"),
            Err(FromStrError::Parse(_))
        ));
    }
}
