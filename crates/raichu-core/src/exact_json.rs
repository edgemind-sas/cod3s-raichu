//! A JSON reader whose floating-point numbers are correctly rounded.
//!
//! `serde_json` parses a float to within one unit in the last place
//! unless its `float_roundtrip` feature is on, and a feature is unified
//! across everything built together: switching it on for one crate would
//! switch it on for every model document the engine reads, which may move
//! a parsed parameter by one ulp and a result with it. The readers of the
//! engine's own result documents (`raichu.exploration`,
//! `raichu.quantification`) parse through this module instead: numbers go
//! through [`str::parse`], which is correctly rounded, so a written
//! document reads back to the same bits, and nothing else in the
//! workspace is touched.

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

/// Parse `text` as one JSON value, numbers correctly rounded. Accepts the
/// documents `serde_json` accepts, except that a member name repeated in
/// one object is refused (`serde_json` keeps the last one silently, which
/// the typed readers built on this module would otherwise inherit), and
/// nesting is capped at 256 levels.
///
/// # Errors
/// [`ParseError`], naming what is wrong and where.
pub fn parse(text: &str) -> Result<Value, ParseError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        text,
        at: 0,
    };
    let value = parser.value(0)?;
    parser.skip_whitespace();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(value)
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

    fn expect(&mut self, literal: &str, value: Value) -> Result<Value, ParseError> {
        if self.text[self.at..].starts_with(literal) {
            self.at += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, ParseError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.skip_whitespace();
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.expect("true", Value::Bool(true)),
            Some(b'f') => self.expect("false", Value::Bool(false)),
            Some(b'n') => self.expect("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
            None => Err(self.error("unexpected end of document")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.at += 1;
        let mut map = Map::new();
        self.skip_whitespace();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_whitespace();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected a member name"));
            }
            let key_at = self.at;
            let key = self.string()?;
            if map.contains_key(&key) {
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
            map.insert(key, value);
            self.skip_whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.error("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, ParseError> {
        self.at += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
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

    fn number(&mut self) -> Result<Value, ParseError> {
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
        let is_float = token.contains(['.', 'e', 'E']);
        if !is_float {
            if let Ok(n) = token.parse::<u64>() {
                return Ok(Value::Number(n.into()));
            }
            if let Ok(n) = token.parse::<i64>() {
                return Ok(Value::Number(n.into()));
            }
        }
        let x: f64 = token.parse().map_err(|e| invalid(&e))?;
        Number::from_f64(x).map(Value::Number).ok_or(ParseError {
            message: format!("number `{token}` is not finite"),
            offset: start,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::parse;

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
}
