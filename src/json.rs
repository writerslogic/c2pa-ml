//! A minimal, dependency-free JSON value model just large enough to read and
//! rewrite a SafeTensors header.
//!
//! Objects preserve key insertion order so re-serializing a header leaves the
//! tensor entries (and their `data_offsets`) exactly where they were. Numbers
//! are kept as their original source text so integer offsets never change
//! representation on a round trip.

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    /// A number, preserved verbatim as its source text.
    Number(String),
    String(String),
    Array(Vec<Value>),
    /// An object with insertion order preserved.
    Object(Vec<(String, Value)>),
}

impl Value {
    pub(crate) fn as_object_mut(&mut self) -> Option<&mut Vec<(String, Value)>> {
        match self {
            Value::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub(crate) fn get<'a>(&'a self, key: &str) -> Option<&'a Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

/// Set (or replace) `key` on an object, preserving the position of an existing
/// key or appending a new one at the end.
pub(crate) fn object_set(entries: &mut Vec<(String, Value)>, key: &str, value: Value) {
    if let Some(slot) = entries.iter_mut().find(|(k, _)| k == key) {
        slot.1 = value;
    } else {
        entries.push((key.to_string(), value));
    }
}

/// Remove `key` from an object if present, returning whether it existed.
pub(crate) fn object_remove(entries: &mut Vec<(String, Value)>, key: &str) -> bool {
    if let Some(i) = entries.iter().position(|(k, _)| k == key) {
        entries.remove(i);
        true
    } else {
        false
    }
}

/// Parse a complete JSON document, rejecting trailing non-whitespace.
pub(crate) fn parse(input: &str) -> Result<Value, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing data at byte {}", p.pos));
    }
    Ok(value)
}

/// Serialize compactly (no insignificant whitespace).
pub(crate) fn to_string(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, v);
            }
            out.push('}');
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected byte {c:#x} at {}", self.pos)),
            None => Err("unexpected end of input".into()),
        }
    }

    fn parse_literal(&mut self, lit: &str, value: Value) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(format!("invalid number at {start}"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "non-UTF-8 number".to_string())?;
        Ok(Value::Number(text.to_string()))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        // Opening quote.
        self.pos += 1;
        let mut s = String::new();
        loop {
            let b = self.peek().ok_or("unterminated string")?;
            self.pos += 1;
            match b {
                b'"' => return Ok(s),
                b'\\' => {
                    let esc = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\u{08}'),
                        b'f' => s.push('\u{0C}'),
                        b'u' => {
                            let cp = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if self.bytes[self.pos..].starts_with(b"\\u") {
                                    self.pos += 2;
                                    let lo = self.parse_hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&lo) {
                                        return Err("invalid low surrogate".into());
                                    }
                                    let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    s.push(char::from_u32(c).ok_or("invalid surrogate pair")?);
                                } else {
                                    return Err("lone high surrogate".into());
                                }
                            } else {
                                s.push(char::from_u32(cp).ok_or("invalid code point")?);
                            }
                        }
                        other => return Err(format!("invalid escape \\{}", other as char)),
                    }
                }
                _ => {
                    // Copy this byte and any UTF-8 continuation bytes verbatim.
                    let start = self.pos - 1;
                    while let Some(nb) = self.peek() {
                        if nb == b'"' || nb == b'\\' {
                            break;
                        }
                        self.pos += 1;
                    }
                    let slice = std::str::from_utf8(&self.bytes[start..self.pos])
                        .map_err(|_| "invalid UTF-8 in string".to_string())?;
                    s.push_str(slice);
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let hex = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or("truncated \\u escape")?;
        let text = std::str::from_utf8(hex).map_err(|_| "invalid \\u escape".to_string())?;
        let cp = u32::from_str_radix(text, 16).map_err(|_| "invalid \\u hex".to_string())?;
        self.pos += 4;
        Ok(cp)
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut entries: Vec<(String, Value)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at {}", self.pos));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_object_order() {
        let src = r#"{"b":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"a":{"dtype":"I64","shape":[1],"data_offsets":[16,24]}}"#;
        let v = parse(src).unwrap();
        assert_eq!(to_string(&v), src);
    }

    #[test]
    fn preserves_number_text() {
        let v = parse("[0,16,1024]").unwrap();
        assert_eq!(to_string(&v), "[0,16,1024]");
    }

    #[test]
    fn parses_escapes_and_unicode() {
        let v = parse(r#""a\"b\né""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "a\"b\né");
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(parse("{} x").is_err());
    }

    #[test]
    fn set_and_remove() {
        let mut v = parse(r#"{"x":"1"}"#).unwrap();
        let obj = v.as_object_mut().unwrap();
        object_set(obj, "y", Value::String("2".into()));
        object_set(obj, "x", Value::String("9".into()));
        assert_eq!(to_string(&v), r#"{"x":"9","y":"2"}"#);
        let obj = v.as_object_mut().unwrap();
        assert!(object_remove(obj, "x"));
        assert!(!object_remove(obj, "z"));
        assert_eq!(to_string(&v), r#"{"y":"2"}"#);
    }
}
