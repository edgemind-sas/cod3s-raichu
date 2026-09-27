//! A JSON reader whose floating-point numbers are correctly rounded.
//!
//! `serde_json` parses a float to within one unit in the last place
//! unless its `float_roundtrip` feature is on, and a feature is unified
//! across everything built together: switching it on for this crate would
//! switch it on for every model document the engine reads, which may move
//! a parsed parameter by one ulp and a result with it. The envelope
//! reader parses through this module instead: numbers go through
//! [`str::parse`], which is correctly rounded, so a written envelope reads
//! back to the same bits, and nothing else in the workspace is touched.

use serde_json::{Map, Number, Value};

/// Parse `text` as one JSON value, numbers correctly rounded.
pub(crate) fn parse(text: &str) -> Result<Value, String> {
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
    fn error(&self, what: &str) -> String {
        format!("{what} at byte {}", self.at)
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.bytes.get(self.at) {
            self.at += 1;
        }
    }

    fn expect(&mut self, literal: &str, value: Value) -> Result<Value, String> {
        if self.text[self.at..].starts_with(literal) {
            self.at += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
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

    fn object(&mut self, depth: usize) -> Result<Value, String> {
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
            let key = self.string()?;
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

    fn array(&mut self, depth: usize) -> Result<Value, String> {
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
    fn string(&mut self) -> Result<String, String> {
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
        serde_json::from_str(&self.text[start..self.at]).map_err(|e| e.to_string())
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.at;
        while let Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') = self.bytes.get(self.at) {
            self.at += 1;
        }
        let token = &self.text[start..self.at];
        // Validate the token as JSON first (Rust's parsers also accept
        // `01`, `+1` or `1.`, which JSON does not).
        serde_json::from_str::<Number>(token).map_err(|e| format!("{e} in number `{token}`"))?;
        let is_float = token.contains(['.', 'e', 'E']);
        if !is_float {
            if let Ok(n) = token.parse::<u64>() {
                return Ok(Value::Number(n.into()));
            }
            if let Ok(n) = token.parse::<i64>() {
                return Ok(Value::Number(n.into()));
            }
        }
        let x: f64 = token
            .parse()
            .map_err(|e| format!("{e} in number `{token}`"))?;
        Number::from_f64(x)
            .map(Value::Number)
            .ok_or_else(|| format!("number `{token}` is not finite"))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::parse;

    /// The case `serde_json` without `float_roundtrip` reads one ulp off.
    #[test]
    fn floats_read_back_to_the_bits_they_were_written_from() {
        let x = 0.497_001_056_589_678_f64;
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
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }
}
