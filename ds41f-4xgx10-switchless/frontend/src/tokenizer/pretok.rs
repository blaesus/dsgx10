//! The DeepSeek-V4.1 pre-tokeniser, hand-written instead of run through a regex engine.
//!
//! tokenizer.json chains three `Split` (behaviour `Isolated`) pre-tokenisers, then a byte-level map:
//!
//! 1. `\p{N}{1,3}`
//! 2. `[一-龥぀-ゟ゠-ヿ]+`
//! 3. ``[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+``
//!
//! Each split runs over every piece the previous one produced; matches and the gaps between them
//! both become pieces. Pattern 3 is matched leftmost-first, as Oniguruma does, and `match_at` below
//! spells out what each alternative's backtracking amounts to. L and M only ever appear together
//! (an M can open the optional prefix of alternative 2, but the match then ends where a run starting
//! at it would), and P and S likewise, so the class table keeps four flags.

use super::unicode_tables::CLASS_RANGES;
use std::sync::OnceLock;

pub const N: u8 = 1;
pub const LM: u8 = 2;
pub const PS: u8 = 4;
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

#[inline]
fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

#[inline]
fn is_ascii_punct(c: char) -> bool {
    c.is_ascii_punctuation()
}

/// Decode the char starting at byte `pos` (which must be a char boundary).
#[inline]
fn char_at(s: &str, pos: usize) -> char {
    // SAFETY-free: slicing at a boundary and taking the first char.
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

/// The end of the match of pattern 3 starting at `pos`, if any alternative matches there.
fn match_at(s: &str, pos: usize) -> Option<usize> {
    let b = s.as_bytes();
    let c0 = char_at(s, pos);
    let f0 = class_of(c0);
    let p1 = pos + c0.len_utf8();
    let c1 = if p1 < b.len() { Some(char_at(s, p1)) } else { None };

    // 1: one ASCII punctuation char, then ASCII letters.
    if c0.is_ascii() && is_ascii_punct(c0) {
        if let Some(c1) = c1 {
            if c1.is_ascii_alphabetic() {
                let mut e = p1;
                while e < b.len() && b[e].is_ascii_alphabetic() {
                    e += 1;
                }
                return Some(e);
            }
        }
    }
    // 2: an optional prefix that is not \r, \n, L, P or S, then a run of L/M.
    if f0 & LM != 0 {
        return Some(run(s, pos, LM));
    }
    if c0 != '\r' && c0 != '\n' && f0 & PS == 0 {
        if let Some(c1) = c1 {
            if class_of(c1) & LM != 0 {
                return Some(run(s, p1, LM));
            }
        }
    }
    // 3: an optional space, a run of P/S, then any \r and \n.
    let ps_start = if f0 & PS != 0 {
        Some(pos)
    } else if c0 == ' ' && c1.map_or(false, |c| class_of(c) & PS != 0) {
        Some(p1)
    } else {
        None
    };
    if let Some(start) = ps_start {
        let mut e = run(s, start, PS);
        while e < b.len() && (b[e] == b'\r' || b[e] == b'\n') {
            e += 1;
        }
        return Some(e);
    }
    // 4-6: whitespace.
    if f0 & WS != 0 {
        let j = run(s, pos, WS);
        // 4: \s*[\r\n]+ ends just after the last \r or \n of the run.
        if let Some(k) = b[pos..j].iter().rposition(|&x| x == b'\r' || x == b'\n') {
            return Some(pos + k + 1);
        }
        // 5: \s+(?!\S): the whole run at the end of the piece, else the run minus its last char.
        if j == b.len() {
            return Some(j);
        }
        let mut last = j - 1;
        while !s.is_char_boundary(last) {
            last -= 1;
        }
        if last > pos {
            return Some(last);
        }
        // 6: \s+
        return Some(j);
    }
    None
}

/// Pattern 3 over one piece: matches and the gaps between them.
fn split3<'a>(s: &'a str, emit: &mut impl FnMut(&'a str)) {
    let b = s.as_bytes();
    let (mut pos, mut last) = (0usize, 0usize);
    while pos < b.len() {
        match match_at(s, pos) {
            Some(end) => {
                if pos > last {
                    emit(&s[last..pos]);
                }
                emit(&s[pos..end]);
                pos = end;
                last = end;
            }
            None => {
                pos += if b[pos] < 0x80 { 1 } else { char_at(s, pos).len_utf8() };
            }
        }
    }
    if last < b.len() {
        emit(&s[last..]);
    }
}

/// Pattern 2 (CJK runs isolated), then pattern 3 on every resulting piece.
fn split2<'a>(s: &'a str, emit: &mut impl FnMut(&'a str)) {
    let mut start = 0usize;
    let mut in_cjk = false;
    for (i, c) in s.char_indices() {
        let cjk = is_cjk(c);
        if cjk != in_cjk {
            if i > start {
                split3(&s[start..i], emit);
            }
            start = i;
            in_cjk = cjk;
        }
    }
    if start < s.len() {
        split3(&s[start..], emit);
    }
}

/// The full pre-tokeniser: calls `emit` for every piece, in order.
pub fn pretokenize<'a>(s: &'a str, emit: &mut impl FnMut(&'a str)) {
    // Pattern 1: runs of \p{N} cut into chunks of at most three chars, each isolated; the text
    // between them goes on to pattern 2.
    let mut gap_start = 0usize;
    let mut chunk_start = 0usize;
    let mut chunk_len = 0usize;
    for (i, c) in s.char_indices() {
        if class_of(c) & N != 0 {
            if chunk_len == 0 {
                if i > gap_start {
                    split2(&s[gap_start..i], emit);
                }
                chunk_start = i;
            } else if chunk_len == 3 {
                split3(&s[chunk_start..i], emit);
                chunk_start = i;
                chunk_len = 0;
            }
            chunk_len += 1;
        } else if chunk_len > 0 {
            split3(&s[chunk_start..i], emit);
            chunk_len = 0;
            gap_start = i;
        }
    }
    if chunk_len > 0 {
        split3(&s[chunk_start..], emit);
    } else if gap_start < s.len() {
        split2(&s[gap_start..], emit);
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
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("x = 12345;\n\n  y"), ["x", " =", " ", "123", "45", ";\n\n", " ", " y"]);
        assert_eq!(pieces("file.py's"), ["file", ".py", "'s"]);
        assert_eq!(pieces("中文abc"), ["中文", "abc"]);
        assert_eq!(pieces("  \n  x"), ["  \n", " ", " x"]);
        assert_eq!(pieces("end  "), ["end", "  "]);
    }
}
