//! The GLM-5.3 pre-tokeniser, hand-written instead of run through a regex engine.
//!
//! tokenizer.json has one `Split` (behaviour `Isolated`) with the Qwen2/GLM-4 pattern, then a
//! byte-level map:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! It is matched leftmost-first, as Oniguruma does, and `match_at` below spells out what each
//! alternative's backtracking amounts to. Every character starts a match of some alternative (it
//! is a letter, a digit, whitespace or "other"), so there are no gaps: the pieces are the matches.
//! The class table keeps three flags: L, N and whitespace; "other" is none of them.

use super::unicode_tables::CLASS_RANGES;
use std::sync::OnceLock;

pub const N: u8 = 1;
pub const L: u8 = 2;
pub const WS: u8 = 8;

fn class_table() -> &'static [u8] {
    static TABLE: OnceLock<Vec<u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0u8; 0x110000];
        for &(a, b, f) in CLASS_RANGES {
            for cp in a..=b {
                t[cp as usize] = f;
            }
        }
        t
    })
}

#[inline]
pub fn class_of(c: char) -> u8 {
    class_table()[c as usize]
}

/// Decode the char starting at byte `pos` (which must be a char boundary).
#[inline]
fn char_at(s: &str, pos: usize) -> char {
    s[pos..].chars().next().unwrap()
}

/// End (byte offset) of the run of chars from `pos` whose class has any bit of `flag`.
#[inline]
fn run(s: &str, mut pos: usize, flag: u8) -> usize {
    let b = s.as_bytes();
    while pos < b.len() {
        let x = b[pos];
        if x < 0x80 {
            if class_of(x as char) & flag == 0 {
                break;
            }
            pos += 1;
        } else {
            let c = char_at(s, pos);
            if class_of(c) & flag == 0 {
                break;
            }
            pos += c.len_utf8();
        }
    }
    pos
}

/// End of the run of "other" chars (`[^\s\p{L}\p{N}]`) from `pos`.
#[inline]
fn run_other(s: &str, mut pos: usize) -> usize {
    let b = s.as_bytes();
    while pos < b.len() {
        let c = if b[pos] < 0x80 { b[pos] as char } else { char_at(s, pos) };
        if class_of(c) != 0 {
            break;
        }
        pos += c.len_utf8();
    }
    pos
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `pos` (the apostrophe): the end of the match. Under Unicode case
/// folding `s` also matches U+017F (long s); no other non-ASCII char folds to these letters.
#[inline]
fn contraction(s: &str, pos: usize) -> Option<usize> {
    let b = s.as_bytes();
    let p = pos + 1;
    let c1 = s[p..].chars().next()?;
    let lower = |c: char| c.to_ascii_lowercase();
    match lower(c1) {
        's' | 't' | 'm' | 'd' => return Some(p + 1),
        '\u{17F}' => return Some(p + 2),
        _ => {}
    }
    let c2 = b.get(p + 1).map(|&x| (x as char).to_ascii_lowercase());
    match (lower(c1), c2) {
        ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l')) => Some(p + 2),
        _ => None,
    }
}

/// The end of the match starting at `pos` (always some alternative matches).
fn match_at(s: &str, pos: usize) -> usize {
    let b = s.as_bytes();
    let c0 = char_at(s, pos);
    let f0 = class_of(c0);
    let p1 = pos + c0.len_utf8();
    let c1 = if p1 < b.len() { Some(char_at(s, p1)) } else { None };
    let f1 = c1.map_or(0xFF, class_of); // 0xFF: no next char

    // 1: contractions
    if c0 == '\'' {
        if let Some(e) = contraction(s, pos) {
            return e;
        }
    }
    // 2: an optional char that is not \r, \n, L or N, then a run of L.
    if f0 & L != 0 {
        return run(s, pos, L);
    }
    if c0 != '\r' && c0 != '\n' && f0 & N == 0 && c1.is_some() && f1 & L != 0 {
        return run(s, p1, L);
    }
    // 3: one to three N.
    if f0 & N != 0 {
        let mut e = p1;
        for _ in 0..2 {
            if e < b.len() {
                let c = char_at(s, e);
                if class_of(c) & N != 0 {
                    e += c.len_utf8();
                    continue;
                }
            }
            break;
        }
        return e;
    }
    // 4: an optional space, a run of "other", then any \r and \n.
    let other_start = if f0 == 0 {
        Some(pos)
    } else if c0 == ' ' && c1.is_some() && f1 == 0 {
        Some(p1)
    } else {
        None
    };
    if let Some(start) = other_start {
        let mut e = run_other(s, start);
        while e < b.len() && (b[e] == b'\r' || b[e] == b'\n') {
            e += 1;
        }
        return e;
    }
    // 5-7: whitespace (f0 is WS here).
    let j = run(s, pos, WS);
    // 5: \s*[\r\n]+ ends just after the last \r or \n of the run.
    if let Some(k) = b[pos..j].iter().rposition(|&x| x == b'\r' || x == b'\n') {
        return pos + k + 1;
    }
    // 6: \s+(?!\S): the whole run at the end of the text, else the run minus its last char.
    if j == b.len() {
        return j;
    }
    let mut last = j - 1;
    while !s.is_char_boundary(last) {
        last -= 1;
    }
    if last > pos {
        return last;
    }
    // 7: \s+
    j
}

/// The full pre-tokeniser: calls `emit` for every piece, in order.
pub fn pretokenize<'a>(s: &'a str, emit: &mut impl FnMut(&'a str)) {
    let mut pos = 0usize;
    while pos < s.len() {
        let end = match_at(s, pos);
        emit(&s[pos..end]);
        pos = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<&str> {
        let mut v = Vec::new();
        pretokenize(s, &mut |p| v.push(p));
        v
    }

    #[test]
    fn basics() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces(" Hello  world\n\n"), [" Hello", " ", " world", "\n\n"]);
        assert_eq!(pieces("12345678"), ["123", "456", "78"]);
        assert_eq!(pieces("don't I'LL"), ["don", "'t", " I", "'LL"]);
        assert_eq!(pieces("\t\tcode()\r\n"), ["\t", "\tcode", "()\r\n"]);
        assert_eq!(pieces("x = 1;\n\n  y"), ["x", " =", " ", "1", ";\n\n", " ", " y"]);
        assert_eq!(pieces("file.py's"), ["file", ".py", "'s"]);
        assert_eq!(pieces("  \n  x"), ["  \n", " ", " x"]);
        assert_eq!(pieces("end  "), ["end", "  "]);
        assert_eq!(pieces("'reason"), ["'re", "ason"]);
        assert_eq!(pieces("'rx"), ["'rx"]);
        assert_eq!(pieces("你好，世界"), ["你好", "，世界"]);
    }
}
