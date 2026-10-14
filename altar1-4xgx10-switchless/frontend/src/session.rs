//! Session matching for exact prefix reuse (docs/protocol.md, "Sessions and exact prefix reuse").
//!
//! Clients do not name their conversation, so a request is matched by text: the session whose text
//! (prompt plus generated output) shares the longest prefix with the new prompt. The shared part is
//! sent as that session's own token ids, which is what is in the engine's KV state; the rest is
//! tokenised on its own from a cut next to an added token, where tokenisation restarts anyway.
//!
//! The table outlives a stop (`save`, `load`; docs/engine.md, "Kept sessions"): the engine keeps its
//! sessions across a planned stop, and the table is what finds them again. It holds the conversations'
//! text, so its file is ours alone: mode 0600 in a directory of mode 0700, written whole or not at all.

use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bytes of shared text below which a match is not worth a session (a BOS and a few tokens).
const MIN_SHARED: usize = 64;
/// A request continues a session in place when the shared text reaches this close to the end of
/// the session's prompt; the encoding may rewrite the tail of the prompt (DeepSeek's rewrote `<think>` as `</think>`; GLM-5.3's and altar-1's do not).
const PROMPT_SLACK: usize = 16;

struct Session {
    text: String,
    ids: Vec<u32>,
    /// ends[i] = byte length of the text of ids[..=i]
    ends: Vec<usize>,
    prompt_len: usize,
    busy: bool,
    last_used: u64,
    /// Unix time of its last request: a saved table leaves out what is older than its limit.
    t_used: u64,
}

pub struct Plan {
    pub session_id: u64,
    /// The session a new session may fork from (0 = none).
    pub base_session_id: u64,
    /// Leading ids taken from a session (the rest is tokenised).
    pub reused: usize,
    pub ids: Vec<u32>,
    /// For the log, counts only: the session whose text shares the most with the prompt (0: none
    /// shares any), the bytes it shares, and the ids its history has in common with `ids`.
    pub matched: u64,
    pub shared: usize,
    pub common: usize,
}

impl Plan {
    /// Where the request starts, for the log (no text, no ids): a continuation of a session, a
    /// fork of one (the engine starts from it if it holds it), or a new session that names none.
    pub fn describe(&self, prompt_bytes: usize) -> String {
        let of = format!("{} of {prompt_bytes} bytes, {} ids in common", self.shared, self.common);
        if self.base_session_id != 0 {
            format!("session {}, a fork of {}: it shares {of}", self.session_id, self.base_session_id)
        } else if self.matched == self.session_id {
            format!("session {} in place: it shares {of}", self.session_id)
        } else if self.matched != 0 {
            format!("session {}, new: the best match, session {}, shares {} bytes (under {MIN_SHARED})", self.session_id, self.matched, self.shared)
        } else {
            format!("session {}, new: no session shares a byte", self.session_id)
        }
    }
}

pub struct SessionTable {
    sessions: HashMap<u64, Session>,
    next_id: u64,
    clock: u64,
    max_sessions: usize,
    max_tokens: usize,
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 64 <= n && a[i..i + 64] == b[i..i + 64] {
        i += 64;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

const TABLE_MAGIC: &[u8; 8] = b"ALTR0001";

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(1469598103934665603u64, |h, &x| (h ^ x as u64).wrapping_mul(1099511628211))
}

/// The directory is ours and nobody else's (made 0700 if it was ours but open to others).
fn private_dir(dir: &Path) -> Result<(), String> {
    let me = std::fs::metadata("/proc/self").map_err(|e| format!("/proc/self: {e}"))?.uid();
    let m = std::fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !m.is_dir() || m.uid() != me {
        return Err(format!("{} is not a directory of ours", dir.display()));
    }
    let abs = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if abs.ancestors().any(|a| a.join(".git").exists()) {
        return Err(format!("{} is in a git tree, where no conversation goes", dir.display()));
    }
    if m.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.b.len() - self.at < n {
            return Err("cut short".into());
        }
        self.at += n;
        Ok(&self.b[self.at - n..self.at])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

impl SessionTable {
    /// Holds up to `max_sessions` sessions and `max_tokens` ids in total. Ids start from the clock,
    /// so that a table begun afresh never reuses an earlier one's ids, whose sessions the engine may
    /// still keep.
    pub fn new(max_sessions: usize, max_tokens: usize) -> Self {
        SessionTable { sessions: HashMap::new(), next_id: (unix_now() << 20).max(1), clock: 0, max_sessions, max_tokens }
    }

    /// The table for the next start: the sessions not in a request and used within `max_age_s`,
    /// written to `path` under a temporary name (mode 0600), synced and renamed into place. The
    /// directory must be ours alone. Returns how many sessions it holds.
    pub fn save(&self, path: &Path, max_age_s: u64) -> Result<usize, String> {
        private_dir(path.parent().unwrap_or(Path::new(".")))?;
        let now = unix_now();
        let (mut body, mut n) = (Vec::new(), 0u64);
        for (&id, s) in &self.sessions {
            if s.busy || now.saturating_sub(s.t_used) > max_age_s {
                continue;
            }
            n += 1;
            for v in [id, s.last_used, s.t_used, s.prompt_len as u64, s.text.len() as u64] {
                body.extend(v.to_le_bytes());
            }
            body.extend(s.text.as_bytes());
            body.extend((s.ids.len() as u64).to_le_bytes());
            for &x in &s.ids {
                body.extend(x.to_le_bytes());
            }
        }
        let mut f = TABLE_MAGIC.to_vec();
        for v in [now, self.next_id, self.clock, n, body.len() as u64, fnv(&body)] {
            f.extend(v.to_le_bytes());
        }
        f.extend(body);
        let tmp = PathBuf::from(format!("{}.new", path.display()));
        let _ = std::fs::remove_file(&tmp);
        let write = || -> std::io::Result<()> {
            let mut o = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
            o.write_all(&f)?;
            o.sync_all()?;
            drop(o);
            std::fs::rename(&tmp, path)?;
            if let Some(d) = path.parent() {
                std::fs::File::open(d)?.sync_all()?;
            }
            Ok(())
        };
        write().map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("{}: {e}", path.display())
        })?;
        Ok(n as usize)
    }

    /// The table an earlier run saved at `path`, when the file is whole, ours alone (mode 0600, in a
    /// directory of ours) and at most `max_age_s` old. The file goes once read: a crash from then on
    /// starts afresh. Sessions not used within `max_age_s`, or whose ids do not spell their text with
    /// this tokenizer, are left out. Err holds the reason ("" when there is no file).
    pub fn load(&mut self, path: &Path, tok: &Tokenizer, max_age_s: u64) -> Result<usize, String> {
        if let Err(e) = std::fs::symlink_metadata(path) {
            return Err(if e.kind() == std::io::ErrorKind::NotFound { String::new() } else { format!("{}: {e}", path.display()) });
        }
        let r = self.load_file(path, tok, max_age_s);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}.new", path.display()));
        r
    }

    fn load_file(&mut self, path: &Path, tok: &Tokenizer, max_age_s: u64) -> Result<usize, String> {
        private_dir(path.parent().unwrap_or(Path::new(".")))?;
        let me = std::fs::metadata("/proc/self").map_err(|e| e.to_string())?.uid();
        let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let m = file.metadata().map_err(|e| e.to_string())?;
        if !m.is_file() || m.uid() != me || m.mode() & 0o777 != 0o600 {
            return Err(format!("{} is not a file of ours of mode 0600", path.display()));
        }
        let mut f = Vec::new();
        file.read_to_end(&mut f).map_err(|e| e.to_string())?;
        if f.len() < 56 || &f[..8] != TABLE_MAGIC {
            return Err("not a saved table".into());
        }
        let mut h = Bytes { b: &f[8..56], at: 0 };
        let (written, next_id, clock, n, body_len, hash) = (h.u64()?, h.u64()?, h.u64()?, h.u64()?, h.u64()?, h.u64()?);
        let now = unix_now();
        if now.saturating_sub(written) > max_age_s {
            return Err(format!("saved {} s ago", now.saturating_sub(written)));
        }
        let body = &f[56..];
        if body.len() as u64 != body_len || fnv(body) != hash {
            return Err("incomplete or damaged".into());
        }
        let mut r = Bytes { b: body, at: 0 };
        let mut got = HashMap::new();
        for _ in 0..n {
            let (id, last_used, t_used, prompt_len, text_len) = (r.u64()?, r.u64()?, r.u64()?, r.u64()? as usize, r.u64()? as usize);
            let text = r.take(text_len)?;
            let n_ids = r.u64()? as usize;
            let raw = r.take(n_ids.checked_mul(4).ok_or("a bad count")?)?;
            let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let Ok(text) = std::str::from_utf8(text) else { continue };
            let vocab = tok.vocab_size() as u32;
            if id == 0 || now.saturating_sub(t_used) > max_age_s || prompt_len > text.len() || ids.iter().any(|&x| x >= vocab) {
                continue;
            }
            let mut ends = Vec::with_capacity(ids.len());
            let mut at = 0;
            for &x in &ids {
                at += tok.token_bytes(x).len();
                ends.push(at);
            }
            if at != text.len() || tok.decode_bytes(&ids) != text.as_bytes() {
                continue; // another tokenizer's ids
            }
            got.insert(id, Session { text: text.to_string(), ids, ends, prompt_len, busy: false, last_used, t_used });
        }
        if r.at != body.len() {
            return Err("a bad body".into());
        }
        let k = got.len();
        self.next_id = self.next_id.max(next_id).max(got.keys().max().map_or(0, |m| m + 1));
        self.clock = self.clock.max(clock);
        self.sessions = got;
        let _ = self.evict(0);
        Ok(k)
    }

    fn new_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Choose the session and the token ids for a prompt. The chosen session is busy until
    /// `finish` or `abandon`. Returns the plan and the sessions to RELEASE on the engine.
    pub fn plan(&mut self, text: &str, tok: &Tokenizer, threads: usize) -> (Plan, Vec<u64>) {
        self.clock += 1;
        let mut top: Option<(u64, usize)> = None;
        for (&id, s) in &self.sessions {
            let n = common_prefix(text.as_bytes(), s.text.as_bytes());
            if n > 0 && top.map_or(true, |(_, bn)| n > bn) {
                top = Some((id, n));
            }
        }
        let best = top.filter(|&(_, n)| n >= MIN_SHARED);
        let (mut plan, cut) = match best {
            None => (Plan { session_id: self.new_id(), base_session_id: 0, reused: 0, ids: Vec::new(), matched: 0, shared: 0, common: 0 }, 0),
            Some((sid, mut shared)) => {
                while !text.is_char_boundary(shared) {
                    shared -= 1;
                }
                let s = &self.sessions[&sid];
                // The last cut point within the shared text that is a token boundary of the session
                // and an edge of an added token in the new text.
                let mut cut = (0, 0); // (byte position, ids before it)
                for (a, e) in tok.added_spans(text) {
                    if a > shared {
                        break;
                    }
                    for p in [a, e] {
                        if p <= shared && p > cut.0 {
                            if let Ok(k) = s.ends.binary_search(&p) {
                                cut = (p, k + 1);
                            }
                        }
                    }
                }
                let ids = s.ids[..cut.1].to_vec();
                let in_place = !s.busy && shared + PROMPT_SLACK >= s.prompt_len;
                let plan = if in_place {
                    Plan { session_id: sid, base_session_id: 0, reused: cut.1, ids, matched: 0, shared: 0, common: 0 }
                } else {
                    Plan { session_id: self.new_id(), base_session_id: sid, reused: cut.1, ids, matched: 0, shared: 0, common: 0 }
                };
                (plan, cut.0)
            }
        };
        plan.ids.extend(tok.encode_parallel(&text[cut..], threads));
        if let Some((sid, n)) = top {
            plan.matched = sid;
            plan.shared = n;
            let s = &self.sessions[&sid].ids;
            plan.common = s.iter().zip(&plan.ids).take_while(|(a, b)| a == b).count();
        }
        let clock = self.clock;
        let s = self.sessions.entry(plan.session_id).or_insert_with(|| Session {
            text: String::new(),
            ids: Vec::new(),
            ends: Vec::new(),
            prompt_len: 0,
            busy: false,
            last_used: clock,
            t_used: 0,
        });
        s.busy = true;
        s.last_used = clock;
        s.t_used = unix_now();
        let released = self.evict(plan.session_id);
        (plan, released)
    }

    /// Record what the session holds after a request: the prompt and the generated ids.
    pub fn finish(&mut self, session_id: u64, prompt_text: &str, prompt_ids: &[u32], generated: &[u32], tok: &Tokenizer) {
        let Some(s) = self.sessions.get_mut(&session_id) else { return };
        let gen_bytes = tok.decode_bytes(generated);
        let mut text = prompt_text.to_string();
        let mut ids = prompt_ids.to_vec();
        // Generated text that is not valid UTF-8 cannot come back verbatim; keep the prompt only.
        if let Ok(g) = std::str::from_utf8(&gen_bytes) {
            text.push_str(g);
            ids.extend_from_slice(generated);
        }
        let mut ends = Vec::with_capacity(ids.len());
        let mut at = 0;
        for &id in &ids {
            at += tok.token_bytes(id).len();
            ends.push(at);
        }
        debug_assert_eq!(at, text.len());
        s.text = text;
        s.ids = ids;
        s.ends = ends;
        s.prompt_len = prompt_text.len();
        s.busy = false;
    }

    /// The request failed: forget the session (the engine's copy is in an unknown state; the
    /// caller releases it).
    pub fn abandon(&mut self, session_id: u64) {
        self.sessions.remove(&session_id);
    }

    /// Drop least recently used idle sessions over the limits; returns their ids.
    fn evict(&mut self, keep: u64) -> Vec<u64> {
        let mut out = Vec::new();
        loop {
            let total: usize = self.sessions.values().map(|s| s.ids.len()).sum();
            if self.sessions.len() <= self.max_sessions && total <= self.max_tokens {
                return out;
            }
            let victim = self.sessions.iter().filter(|(&id, s)| id != keep && !s.busy).min_by_key(|(_, s)| s.last_used).map(|(&id, _)| id);
            match victim {
                Some(id) => {
                    self.sessions.remove(&id);
                    out.push(id);
                }
                None => return out,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: &str = "/data/hf/hub/models--nvidia--GLM-5.3-NVFP4/snapshots/e3b8f9f2fad29727dfa95e766c5cac18ea1414bf/tokenizer.json";

    #[test]
    fn reuses_the_models_own_ids() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let mut t = SessionTable::new(8, 1 << 20);
        let sys = "[gMASK]<sop><|system|>You are a careful assistant who answers briefly and well.";
        let p1 = format!("{sys}<|user|>hello<|assistant|><think>");
        let (plan, _) = t.plan(&p1, &tok, 1);
        assert_eq!(plan.ids, tok.encode(&p1));
        // the model spelled "Hi there" in single letters, which BPE would never produce
        let mut gen: Vec<u32> = "Hi there".chars().map(|c| tok.encode(&c.to_string())[0]).collect();
        gen.push(tok.token_id("<|user|>").unwrap());
        t.finish(plan.session_id, &p1, &plan.ids, &gen, &tok);
        let p2 = format!("{p1}Hi there<|user|>next<|assistant|><think>");
        let (plan2, _) = t.plan(&p2, &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        let mut want = plan.ids.clone();
        want.extend(&gen);
        // the stop id the model ended with is the next turn's `<|user|>`: kept too
        want.extend(tok.encode("next<|assistant|><think>"));
        assert_eq!(plan2.ids, want);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // the same system prompt in another conversation forks instead of taking over the session
        let p3 = format!("{sys}<|user|>something else entirely<|assistant|><think>");
        let (plan3, _) = t.plan(&p3, &tok, 1);
        assert_ne!(plan3.session_id, plan.session_id);
        assert_eq!(plan3.base_session_id, plan.session_id);
        assert_eq!(tok.decode(&plan3.ids), p3);
    }

    #[test]
    fn the_plan_says_where_it_starts() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let mut t = SessionTable::new(8, 1 << 20);
        // pi-like: a long system text that ends with the time, then the tools and the first message
        let body = "You are an expert coding assistant. Read files, run commands and edit code.\n".repeat(40);
        let conv = |time: &str, msg: &str| {
            format!("[gMASK]<sop><|system|>{body}Current date and time: Monday, October 12, 2026 at {time}\n\n## Tools\n\nnone<|user|>{msg}<|assistant|><think>")
        };
        let a = conv("09:14:08 AM", "fix the build");
        let (pa, _) = t.plan(&a, &tok, 1);
        assert_eq!(pa.describe(a.len()), format!("session {}, new: no session shares a byte", pa.session_id));
        t.finish(pa.session_id, &a, &pa.ids, &tok.encode("Done."), &tok);
        // another conversation, a minute later: a fork of the first, which has every id before the time
        let b = conv("09:15:31 AM", "write the tests");
        let (pb, _) = t.plan(&b, &tok, 1);
        assert_eq!(pb.base_session_id, pa.session_id);
        assert_eq!(pb.ids, tok.encode(&b), "the ids are a cold run's");
        let head = tok.encode(&b[..b.find("09:15").unwrap()]).len();
        assert!(pb.common + 2 >= head && pb.common <= head + 2, "{} ids in common, {head} before the time", pb.common);
        let d = pb.describe(b.len());
        assert!(d.starts_with(&format!("session {}, a fork of {}: it shares ", pb.session_id, pa.session_id)), "{d}");
        assert!(d.ends_with(&format!("of {} bytes, {} ids in common", b.len(), pb.common)), "{d}");
        // the next turn of the first: in place
        let a2 = format!("{a}Done.<|user|>thanks<|assistant|><think>");
        let (pa2, _) = t.plan(&a2, &tok, 1);
        assert_eq!(pa2.session_id, pa.session_id);
        assert!(pa2.describe(a2.len()).starts_with(&format!("session {} in place: it shares ", pa.session_id)));
        t.finish(pa2.session_id, &a2, &pa2.ids, &tok.encode("Sure."), &tok);
        // a prompt that shares only the BOS and a few bytes: a new session, which names no base
        let c = "[gMASK]<sop><|user|>hi<|assistant|><think>";
        let (pc, _) = t.plan(c, &tok, 1);
        assert_eq!(pc.base_session_id, 0);
        let n = common_prefix(c.as_bytes(), a.as_bytes());
        assert!(n > 0 && n < MIN_SHARED);
        let d = pc.describe(c.len());
        assert!(d.starts_with(&format!("session {}, new: the best match, session ", pc.session_id)), "{d}");
        assert!(d.ends_with(&format!(", shares {n} bytes (under 64)")), "{d}");
    }

    #[test]
    fn a_saved_table_finds_its_sessions_again() {
        let Ok(tok) = Tokenizer::from_file(TOKENIZER) else { return };
        let dir = std::env::temp_dir().join(format!("altarfront-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("front.table");
        let mut t = SessionTable::new(8, 1 << 20);
        let p1 = "[gMASK]<sop><|system|>You are a careful assistant who answers briefly and well.<|user|>hello<|assistant|><think>";
        let (plan, _) = t.plan(p1, &tok, 1);
        let gen = tok.encode("Hi there.");
        t.finish(plan.session_id, p1, &plan.ids, &gen, &tok);
        let (busy, _) = t.plan("[gMASK]<sop><|system|>Another conversation, still in its request, which is not saved.", &tok, 1);
        let repo = dir.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir(repo.join("run")).unwrap();
        let e = t.save(&repo.join("run/front.table"), 3600).unwrap_err();
        assert!(e.contains("git tree"), "{e}");
        assert!(!repo.join("run/front.table").exists() && !repo.join("run/front.table.new").exists());
        std::fs::remove_dir_all(&repo).unwrap();
        assert_eq!(t.save(&path, 3600), Ok(1));
        let m = std::fs::metadata(&path).unwrap();
        assert_eq!(m.mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&dir).unwrap().mode() & 0o777, 0o700);
        let mut u = SessionTable::new(8, 1 << 20);
        assert_eq!(u.load(&path, &tok, 3600), Ok(1));
        assert!(!path.exists(), "the file goes once read");
        assert!(u.next_id > busy.session_id, "ids go on after the saved table's");
        let p2 = format!("{p1}Hi there.<|user|>next<|assistant|><think>");
        let (plan2, _) = u.plan(&p2, &tok, 1);
        assert_eq!(plan2.session_id, plan.session_id);
        assert_eq!(plan2.reused, plan.ids.len() + gen.len());
        assert_eq!(tok.decode(&plan2.ids), p2);
        // nothing there, a damaged file, a file open to others: nothing taken, the file gone
        assert_eq!(u.load(&path, &tok, 3600), Err(String::new()));
        t.save(&path, 3600).unwrap();
        let mut b = std::fs::read(&path).unwrap();
        let k = b.len() - 3;
        b[k] ^= 1;
        std::fs::write(&path, &b).unwrap();
        assert_eq!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600), Err("incomplete or damaged".into()));
        assert!(!path.exists());
        t.save(&path, 3600).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600).is_err());
        // a table saved long ago, a session not used within the limit: left out
        t.save(&path, 3600).unwrap();
        let mut b = std::fs::read(&path).unwrap();
        b[8..16].copy_from_slice(&1u64.to_le_bytes());
        std::fs::write(&path, &b).unwrap();
        assert!(SessionTable::new(8, 1 << 20).load(&path, &tok, 3600).unwrap_err().starts_with("saved"));
        t.sessions.get_mut(&plan.session_id).unwrap().t_used = 1;
        assert_eq!(t.save(&path, 3600), Ok(0));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
