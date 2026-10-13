//! Stop strings and the repetition-loop guard, applied to the generated stream.

/// Finds the client's stop strings in streamed text, holding back a tail that could be the start
/// of one. The stop string itself is not passed on.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new() }
    }

    /// Text that can be passed on, and whether a stop string ended it.
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.buf.push_str(s);
        if let Some(p) = self.stops.iter().filter_map(|st| self.buf.find(st.as_str())).min() {
            let out = self.buf[..p].to_string();
            self.buf.clear();
            return (out, true);
        }
        let mut keep = 0;
        for st in &self.stops {
            for k in (1..st.len().min(self.buf.len() + 1)).rev() {
                let at = self.buf.len() - k;
                if self.buf.is_char_boundary(at) && st.starts_with(&self.buf[at..]) {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let n = self.buf.len() - keep;
        let out = self.buf[..n].to_string();
        self.buf.drain(..n);
        (out, false)
    }

    /// End of the stream: the held-back tail is ordinary text after all.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

/// The kit's loop abort (adapter/loop_abort.py), same defaults: stop when the last `repeats`
/// copies of an n-gram of `ngram..=ngram_max` tokens are identical, when one token repeats
/// `identical` times, or when the last `line_repeats` non-empty lines (at least `line_min` chars)
/// of the last 512 tokens' text are identical.
#[derive(Clone, Debug)]
pub struct LoopGuard {
    pub ngram: usize,
    pub ngram_max: usize,
    pub repeats: usize,
    pub identical: usize,
    pub line_repeats: usize,
    pub line_min: usize,
}

pub const LINE_WINDOW_TOKENS: usize = 512;

impl Default for LoopGuard {
    fn default() -> Self {
        LoopGuard { ngram: 32, ngram_max: 256, repeats: 4, identical: 64, line_repeats: 8, line_min: 16 }
    }
}

/// Python's str.splitlines boundaries.
fn is_line_break(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

fn py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

impl LoopGuard {
    /// The number of tokens to keep if the output is looping (one copy of the cycle, or one token
    /// of a run; all of it for repeated lines), else None. `tail_text` is the text of the last
    /// LINE_WINDOW_TOKENS ids.
    pub fn check(&self, ids: &[u32], tail_text: &str) -> Option<usize> {
        let n = ids.len();
        if n == 0 {
            return None;
        }
        if self.identical > 0 && n >= self.identical {
            let last = ids[n - 1];
            let run = ids.iter().rev().take_while(|&&x| x == last).count();
            if run >= self.identical {
                return Some(n - run + 1);
            }
        }
        if self.ngram > 0 && self.repeats > 0 {
            let max_p = self.ngram_max.max(self.ngram).min(n / self.repeats);
            for period in self.ngram..=max_p {
                let unit = &ids[n - period..];
                if (1..self.repeats).all(|r| &ids[n - (r + 1) * period..n - r * period] == unit) {
                    return Some(n - (self.repeats - 1) * period);
                }
            }
        }
        if self.line_repeats > 0 && self.line_min > 0 {
            let lines: Vec<&str> =
                tail_text.split(is_line_break).filter(|l| !l.trim_matches(py_space).is_empty()).map(|l| l.trim_end_matches(py_space)).collect();
            if lines.len() >= self.line_repeats {
                let last = lines[lines.len() - 1];
                if last.chars().count() >= self.line_min && lines[lines.len() - self.line_repeats..].iter().all(|l| *l == last) {
                    return Some(n);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings() {
        let mut m = StopMatcher::new(vec!["STOP".into(), "\n\n".into()]);
        let (a, s1) = m.push("hello ST");
        assert_eq!((a.as_str(), s1), ("hello ", false));
        let (b, s2) = m.push("ORM");
        assert_eq!((b.as_str(), s2), ("STORM", false));
        let (c, s3) = m.push("x\n");
        assert_eq!((c.as_str(), s3), ("x", false));
        let (d, s4) = m.push("\nrest");
        assert_eq!((d.as_str(), s4), ("", true));
    }

    #[test]
    fn loops() {
        let g = LoopGuard::default();
        let cycle: Vec<u32> = (0..6).collect();
        let ids: Vec<u32> = cycle.iter().cycle().take(36 * 4).cloned().collect();
        assert_eq!(g.check(&ids, ""), Some(36));
        assert_eq!(g.check(&ids[..36 * 4 - 1], ""), None);
        assert_eq!(g.check(&vec![7; 64], ""), Some(1));
        let text = "line that repeats itself\n".repeat(8);
        assert_eq!(g.check(&[1, 2, 3], &text), Some(3));
        assert_eq!(g.check(&[1, 2, 3], &"short\n".repeat(8)), None);
    }
}
