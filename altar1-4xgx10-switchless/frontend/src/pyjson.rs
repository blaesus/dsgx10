//! `json.dumps(value, ensure_ascii=False)` exactly as Python writes it, and Python's `str()` of a
//! JSON value where the publisher's encoder formats one into a template.
//!
//! The chat encoding renders tool schemas and tool-call arguments with Python's json module, so the
//! prompt text (and with it the KV cache) depends on its formatting: `", "` and `": "` separators,
//! keys in input order, non-ASCII characters left as they are, integers verbatim and floats in
//! `repr` form (`1.0`, `1e-05`, `1e+16`). Values are parsed with serde_json's `arbitrary_precision`
//! (numbers keep their source text) and `preserve_order` (objects keep key order), so every
//! distinction Python's parser makes survives.

use serde_json::{Number, Value};

/// Python `json.dumps(v, ensure_ascii=False)`.
pub fn dumps(v: &Value) -> String {
    let mut s = String::new();
    write_value(&mut s, v);
    s
}

fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_repr(n)),
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, x);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, x);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
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

/// Whether Python's json module would read this number literal as an int.
fn is_int_literal(s: &str) -> bool {
    !s.contains(['.', 'e', 'E'])
}

/// Python's `json.loads(s)`, or None where it raises. Unlike serde_json it takes the constants
/// `NaN`, `Infinity` and `-Infinity` (kept as number literals that `number_repr` writes back the way
/// Python does). Strings with unpaired surrogates, which Python keeps and Rust cannot, count as
/// failures.
pub fn loads(s: &str) -> Option<Value> {
    let mut p = Loader { b: s.as_bytes(), s, i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    (p.i == p.b.len()).then_some(v)
}

struct Loader<'a> {
    b: &'a [u8],
    s: &'a str,
    i: usize,
    depth: usize,
}

impl<'a> Loader<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str) -> bool {
        if self.s[self.i..].starts_with(word) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn num(text: &str) -> Value {
        Value::Number(Number::from_string_unchecked(text.to_string()))
    }

    fn value(&mut self) -> Option<Value> {
        self.depth += 1;
        if self.depth > 900 {
            return None; // Python's recursion limit
        }
        let v = match *self.b.get(self.i)? {
            b'"' => Value::String(self.string()?),
            b'{' => {
                self.i += 1;
                let mut m = serde_json::Map::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                } else {
                    loop {
                        if self.b.get(self.i) != Some(&b'"') {
                            return None;
                        }
                        let k = self.string()?;
                        self.ws();
                        if self.b.get(self.i) != Some(&b':') {
                            return None;
                        }
                        self.i += 1;
                        self.ws();
                        let v = self.value()?;
                        m.insert(k, v); // a repeated key keeps its first place and takes the last value
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => {
                                self.i += 1;
                                self.ws();
                            }
                            Some(b'}') => {
                                self.i += 1;
                                break;
                            }
                            _ => return None,
                        }
                    }
                }
                Value::Object(m)
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                } else {
                    loop {
                        a.push(self.value()?);
                        self.ws();
                        match self.b.get(self.i) {
                            Some(b',') => {
                                self.i += 1;
                                self.ws();
                            }
                            Some(b']') => {
                                self.i += 1;
                                break;
                            }
                            _ => return None,
                        }
                    }
                }
                Value::Array(a)
            }
            b'n' if self.lit("null") => Value::Null,
            b't' if self.lit("true") => Value::Bool(true),
            b'f' if self.lit("false") => Value::Bool(false),
            b'N' if self.lit("NaN") => Self::num("NaN"),
            b'I' if self.lit("Infinity") => Self::num("Infinity"),
            b'-' if self.lit("-Infinity") => Self::num("-Infinity"),
            b'-' | b'0'..=b'9' => {
                // -?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?
                let start = self.i;
                let digits = |l: &mut Self| {
                    let st = l.i;
                    while l.i < l.b.len() && l.b[l.i].is_ascii_digit() {
                        l.i += 1;
                    }
                    l.i - st
                };
                if self.b[self.i] == b'-' {
                    self.i += 1;
                }
                match self.b.get(self.i) {
                    Some(b'0') => self.i += 1,
                    Some(b'1'..=b'9') => {
                        digits(self);
                    }
                    _ => return None,
                }
                if self.b.get(self.i) == Some(&b'.') && self.b.get(self.i + 1).map_or(false, u8::is_ascii_digit) {
                    self.i += 1;
                    digits(self);
                }
                if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
                    let save = self.i;
                    self.i += 1;
                    if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                        self.i += 1;
                    }
                    if digits(self) == 0 {
                        self.i = save;
                    }
                }
                Self::num(&self.s[start..self.i])
            }
            _ => return None,
        };
        self.depth -= 1;
        Some(v)
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.s.get(self.i..self.i + 4)?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        self.i += 4;
        u32::from_str_radix(h, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && !matches!(self.b[self.i], b'"' | b'\\') && self.b[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(&self.s[start..self.i]);
            match *self.b.get(self.i)? {
                b'"' => {
                    self.i += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.i += 1;
                    let c = *self.b.get(self.i)?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) && self.s[self.i..].starts_with("\\u") {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4() {
                                    Some(lo) if (0xDC00..0xE000).contains(&lo) => u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00),
                                    _ => self.i = save,
                                }
                            }
                            out.push(char::from_u32(u)?); // None for an unpaired surrogate
                        }
                        _ => return None,
                    }
                }
                _ => return None, // control character (strict mode)
            }
        }
    }
}

/// The text Python writes for a JSON number it read: ints verbatim (`-0` becomes `0`), floats via repr.
pub fn number_repr(n: &Number) -> String {
    let s = n.to_string();
    if matches!(s.as_str(), "NaN" | "Infinity" | "-Infinity") {
        return s;
    }
    if is_int_literal(&s) {
        let neg = s.starts_with('-');
        let digits = s.trim_start_matches('-').trim_start_matches('0');
        if digits.is_empty() {
            return "0".to_string();
        }
        return if neg { format!("-{digits}") } else { digits.to_string() };
    }
    match s.parse::<f64>() {
        Ok(f) => float_repr(f),
        Err(_) => s,
    }
}

/// Python's `repr(float)`: the shortest round-tripping digits, fixed notation for exponents in
/// [-4, 16), scientific with a signed two-digit-minimum exponent otherwise; `.0` on integral values.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    // Rust's {:e} prints the shortest round-tripping digits: "d.ddde[-]x".
    let e = format!("{:e}", f.abs());
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.DIGITS * 10^decpt
    let sign = if f < 0.0 { "-" } else { "" };
    if decpt > 16 || decpt < -3 {
        let mut m = digits[..1].to_string();
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        let x = decpt - 1;
        format!("{sign}{m}e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!("{sign}{digits}{}.0", "0".repeat(decpt as usize - digits.len()))
    } else {
        format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    }
}

/// Python's `str(x)` for a JSON value placed into a `str.format` template. Strings are themselves;
/// `None`, `True`, `False` and numbers as Python prints them. Lists and dicts get Python's repr in the
/// common cases (quoted strings without escapes); the publisher's code only does this for malformed
/// messages.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => py_repr(v),
    }
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => number_repr(n),
        Value::String(s) => {
            if s.contains('\'') && !s.contains('"') {
                format!("\"{s}\"")
            } else {
                format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
            }
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, x)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(x))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => {
            let s = n.to_string();
            s.parse::<f64>().map(|f| f != 0.0).unwrap_or(true)
        }
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_like_python() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789.125, "123456789.125"),
            (-2.5e-7, "-2.5e-07"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (0.30000000000000004, "0.30000000000000004"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn loads_like_python() {
        let v = loads(" {\"a\": NaN, \"b\": [-Infinity, 1E2, -0], \"a\": 2, \"s\": \"\\ud83d\\ude00\\u00e9\"} ").unwrap();
        assert_eq!(dumps(&v), "{\"a\": 2, \"b\": [-Infinity, 100.0, 0], \"s\": \"😀é\"}");
        for bad in ["", "01", "1.", "[1,]", "{\"a\":1,}", "\"\\ud800\"", "\"a\nb\"", "-NaN", "nan", "1 2", "\u{feff}1"] {
            assert!(loads(bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn dumps_like_python() {
        let v: Value = serde_json::from_str(r#"{"b": [1, 2.50, -0, 1E2, true, null], "a": "é\n\"\u0001", "n": 12345678901234567890123}"#).unwrap();
        assert_eq!(dumps(&v), "{\"b\": [1, 2.5, 0, 100.0, true, null], \"a\": \"é\\n\\\"\\u0001\", \"n\": 12345678901234567890123}");
    }
}
