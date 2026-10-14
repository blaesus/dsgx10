//! Byte-level BPE tokeniser for GLM-5.3, read from the checkpoint's tokenizer.json.
//!
//! Encoding follows the Hugging Face `tokenizers` pipeline the reference uses:
//! 1. added tokens are cut out of the text first, leftmost-longest, in two passes (tokens with
//!    `normalized: false`, then those with `normalized: true` on what is left; there is no
//!    normaliser, so both passes see the raw text; GLM-5.3's 36 are all `normalized: false`);
//! 2. the rest is pre-tokenised (pretok.rs) into pieces;
//! 3. with `ignore_merges` (GLM-5.3 sets it), a piece that is itself a vocabulary entry is that one
//!    id; otherwise its bytes go through BPE: repeatedly merge the adjacent pair with the lowest
//!    merge rank, leftmost first.
//!
//! Merges are read in either form tokenizers writes: `"a b"` strings (older files) or `["a", "b"]`
//! arrays (GLM-5.3's).
//! Decoding maps every id back to its bytes (added tokens to their text).

mod pretok;
mod unicode_tables;

pub use pretok::pretokenize;

use serde::Deserialize;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;
use std::hash::{BuildHasherDefault, Hasher};

/// A small, fast hasher for integer and short byte keys (FxHash).
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf) ^ ((rest.len() as u64) << 56));
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(FX_SEED);
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuild>;

/// GPT-2's byte → printable char map, which byte-level vocab strings are written in.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

#[derive(Deserialize)]
struct JsonAddedToken {
    id: u32,
    content: String,
    special: bool,
    normalized: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonMerge {
    Pair(String, String),
    Joined(String),
}

#[derive(Deserialize)]
struct JsonModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<JsonMerge>,
    #[serde(default)]
    ignore_merges: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    dropout: Option<f64>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
}

#[derive(Deserialize)]
struct JsonTokenizer {
    added_tokens: Vec<JsonAddedToken>,
    model: JsonModel,
}

/// A byte trie over added-token strings, for leftmost-longest matching.
struct Trie {
    // nodes[i] = (sorted children (byte, node), token id ending here)
    nodes: Vec<(Vec<(u8, u32)>, Option<u32>)>,
    first: [bool; 256],
}

impl Trie {
    fn new() -> Self {
        Trie { nodes: vec![(Vec::new(), None)], first: [false; 256] }
    }
    fn insert(&mut self, s: &[u8], id: u32) {
        self.first[s[0] as usize] = true;
        let mut n = 0usize;
        for &b in s {
            let next = match self.nodes[n].0.binary_search_by_key(&b, |&(k, _)| k) {
                Ok(i) => self.nodes[n].0[i].1 as usize,
                Err(i) => {
                    let new = self.nodes.len();
                    self.nodes.push((Vec::new(), None));
                    self.nodes[n].0.insert(i, (b, new as u32));
                    new
                }
            };
            n = next;
        }
        self.nodes[n].1 = Some(id);
    }
    /// Longest token starting at `pos`: (end, id).
    fn longest_at(&self, b: &[u8], pos: usize) -> Option<(usize, u32)> {
        let mut n = 0usize;
        let mut best = None;
        let mut i = pos;
        while i < b.len() {
            match self.nodes[n].0.binary_search_by_key(&b[i], |&(k, _)| k) {
                Ok(j) => n = self.nodes[n].0[j].1 as usize,
                Err(_) => break,
            }
            i += 1;
            if let Some(id) = self.nodes[n].1 {
                best = Some((i, id));
            }
        }
        best
    }
    /// The same split by byte positions within `b[start..end]`: `gap(a, b)` and `tok(a, b)`.
    fn split_pos(&self, b: &[u8], start: usize, end: usize, gap: &mut impl FnMut(usize, usize), tok: &mut impl FnMut(usize, usize)) {
        let (mut i, mut last) = (start, start);
        while i < end {
            if self.first[b[i] as usize] {
                if let Some((e, _)) = self.longest_at(&b[..end], i) {
                    if i > last {
                        gap(last, i);
                    }
                    tok(i, e);
                    i = e;
                    last = e;
                    continue;
                }
            }
            i += 1;
        }
        if last < end {
            gap(last, end);
        }
    }
    /// Leftmost-longest split of `s`: calls `text` for gaps and `tok` for matches.
    fn split<'a>(&self, s: &'a str, text: &mut impl FnMut(&'a str), tok: &mut impl FnMut(u32)) {
        let b = s.as_bytes();
        let (mut i, mut last) = (0usize, 0usize);
        while i < b.len() {
            if self.first[b[i] as usize] {
                if let Some((end, id)) = self.longest_at(b, i) {
                    if i > last {
                        text(&s[last..i]);
                    }
                    tok(id);
                    i = end;
                    last = end;
                    continue;
                }
            }
            i += 1;
        }
        if last < b.len() {
            text(&s[last..]);
        }
    }
}

pub struct Tokenizer {
    /// merges[(left, right)] = (rank, merged id)
    merges: FxHashMap<u64, (u32, u32)>,
    /// id of the single-byte token for every byte
    byte_ids: [u32; 256],
    /// bytes of every id
    id_bytes: Vec<Box<[u8]>>,
    added_raw: Trie,
    added_norm: Trie,
    special: Vec<bool>,
    token_ids: HashMap<String, u32>,
    /// With `ignore_merges`: every vocabulary entry by its bytes (a piece found here is one id).
    whole: Option<FxHashMap<Box<[u8]>, u32>>,
    /// The largest vocabulary entry, in bytes (longer pieces skip the `whole` lookup).
    whole_max: usize,
}

/// Per-thread memo of piece → ids.
pub struct Cache {
    map: FxHashMap<Box<[u8]>, Box<[u32]>>,
}

impl Default for Cache {
    fn default() -> Self {
        Cache { map: FxHashMap::default() }
    }
}

const CACHE_MAX: usize = 200_000;
const CACHE_MAX_PIECE: usize = 64;

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let j: JsonTokenizer = serde_json::from_str(text).map_err(|e| format!("tokenizer.json: {e}"))?;
        if j.model.kind != "BPE" {
            return Err(format!("unsupported model type {}", j.model.kind));
        }
        if j.model.byte_fallback || j.model.dropout.is_some_and(|d| d > 0.0) {
            return Err("byte_fallback and dropout are not supported".into());
        }
        if j.model.continuing_subword_prefix.as_deref().is_some_and(|s| !s.is_empty())
            || j.model.end_of_word_suffix.as_deref().is_some_and(|s| !s.is_empty())
        {
            return Err("continuing_subword_prefix and end_of_word_suffix are not supported".into());
        }
        let b2u = bytes_to_unicode();
        let mut u2b: HashMap<char, u8> = HashMap::new();
        for (b, c) in b2u.iter().enumerate() {
            u2b.insert(*c, b as u8);
        }
        let max_id = j
            .model
            .vocab
            .values()
            .copied()
            .chain(j.added_tokens.iter().map(|t| t.id))
            .max()
            .unwrap_or(0) as usize;
        let mut id_bytes: Vec<Box<[u8]>> = vec![Box::new([]); max_id + 1];
        let mut token_ids = HashMap::new();
        let mut whole: Option<FxHashMap<Box<[u8]>, u32>> = j.model.ignore_merges.then(FxHashMap::default);
        let mut whole_max = 0;
        for (s, &id) in &j.model.vocab {
            // Vocab strings are byte-level encoded; special strings stored in the vocab (if any)
            // are not, and are overwritten below by their added-token text.
            let bytes: Option<Vec<u8>> = s.chars().map(|c| u2b.get(&c).copied()).collect();
            if let (Some(w), Some(b)) = (whole.as_mut(), bytes.as_ref()) {
                whole_max = whole_max.max(b.len());
                w.insert(b.clone().into_boxed_slice(), id);
            }
            id_bytes[id as usize] = bytes.unwrap_or_else(|| s.as_bytes().to_vec()).into_boxed_slice();
            token_ids.insert(s.clone(), id);
        }
        let mut special = vec![false; max_id + 1];
        let mut added_raw = Trie::new();
        let mut added_norm = Trie::new();
        for t in &j.added_tokens {
            id_bytes[t.id as usize] = t.content.as_bytes().to_vec().into_boxed_slice();
            special[t.id as usize] = t.special;
            token_ids.insert(t.content.clone(), t.id);
            if t.normalized {
                added_norm.insert(t.content.as_bytes(), t.id);
            } else {
                added_raw.insert(t.content.as_bytes(), t.id);
            }
        }
        let mut byte_ids = [0u32; 256];
        for b in 0..256 {
            let s = b2u[b].to_string();
            byte_ids[b] = *j.model.vocab.get(&s).ok_or_else(|| format!("byte {b} has no token"))?;
        }
        let mut merges = FxHashMap::default();
        merges.reserve(j.model.merges.len());
        for (rank, m) in j.model.merges.iter().enumerate() {
            let (a, b) = match m {
                JsonMerge::Pair(a, b) => (a.as_str(), b.as_str()),
                JsonMerge::Joined(m) => m.split_once(' ').ok_or_else(|| format!("bad merge {m:?}"))?,
            };
            let ia = *j.model.vocab.get(a).ok_or_else(|| format!("merge part {a:?} not in vocab"))?;
            let ib = *j.model.vocab.get(b).ok_or_else(|| format!("merge part {b:?} not in vocab"))?;
            let ab = format!("{a}{b}");
            let iab = *j.model.vocab.get(&ab).ok_or_else(|| format!("merge result {ab:?} not in vocab"))?;
            merges.entry(pair_key(ia, ib)).or_insert((rank as u32, iab));
        }
        Ok(Tokenizer { merges, byte_ids, id_bytes, added_raw, added_norm, special, token_ids, whole, whole_max })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_bytes.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_ids.get(s).copied()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.id_bytes.get(id as usize).map(|b| &b[..]).unwrap_or(&[])
    }

    /// Encode `text`, cutting out added tokens (special or not) wherever they occur.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut cache = Cache::default();
        let mut out = Vec::with_capacity(text.len() / 3 + 16);
        self.encode_with(text, &mut cache, &mut out);
        out
    }

    /// Encode with a caller-owned cache, appending to `out`.
    pub fn encode_with(&self, text: &str, cache: &mut Cache, out: &mut Vec<u32>) {
        // Two passes over added tokens, then pre-tokenise and BPE what is left.
        let out_cell = std::cell::RefCell::new(out);
        let mut plain = |seg: &str| {
            let mut emit = |t: &str| {
                pretokenize(t, &mut |piece| self.bpe_piece(piece.as_bytes(), cache, &mut out_cell.borrow_mut()));
            };
            self.added_norm.split(seg, &mut emit, &mut |id| out_cell.borrow_mut().push(id));
        };
        self.added_raw.split(text, &mut plain, &mut |id| out_cell.borrow_mut().push(id));
    }

    /// Encode a long text on several threads. Cuts only where a piece boundary is certain:
    /// after a newline that is followed by a non-whitespace char.
    pub fn encode_parallel(&self, text: &str, threads: usize) -> Vec<u32> {
        const CHUNK: usize = 1 << 16;
        if text.len() < 2 * CHUNK || threads <= 1 {
            return self.encode(text);
        }
        let target = (text.len() / threads).max(CHUNK);
        let b = text.as_bytes();
        let mut cuts = vec![0usize];
        let mut pos = target;
        while pos < b.len() {
            // find '\n' followed by a byte that starts a non-whitespace char
            let mut i = pos;
            let mut found = None;
            while i + 1 < b.len() {
                if b[i] == b'\n' {
                    let nb = b[i + 1];
                    let ws = matches!(nb, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C) || nb >= 0x80;
                    if !ws && self.cut_is_safe(text, i + 1) {
                        found = Some(i + 1);
                        break;
                    }
                }
                i += 1;
            }
            match found {
                Some(c) => {
                    cuts.push(c);
                    pos = c + target;
                }
                None => break,
            }
        }
        cuts.push(b.len());
        let mut parts: Vec<Vec<u32>> = vec![Vec::new(); cuts.len() - 1];
        std::thread::scope(|sc| {
            for (k, part) in parts.iter_mut().enumerate() {
                let seg = &text[cuts[k]..cuts[k + 1]];
                sc.spawn(move || {
                    let mut cache = Cache::default();
                    self.encode_with(seg, &mut cache, part);
                });
            }
        });
        parts.concat()
    }

    /// A cut is safe if no added token spans it.
    fn cut_is_safe(&self, text: &str, cut: usize) -> bool {
        // Added tokens are at most a few dozen bytes; check whether one starting shortly before the
        // cut would cross it.
        let b = text.as_bytes();
        let lo = cut.saturating_sub(64);
        for start in lo..cut {
            for trie in [&self.added_raw, &self.added_norm] {
                if trie.first[b[start] as usize] {
                    if let Some((end, _)) = trie.longest_at(b, start) {
                        if end > cut {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn bpe_piece(&self, piece: &[u8], cache: &mut Cache, out: &mut Vec<u32>) {
        if piece.len() == 1 {
            out.push(self.byte_ids[piece[0] as usize]);
            return;
        }
        if let Some(w) = &self.whole {
            if piece.len() <= self.whole_max {
                if let Some(&id) = w.get(piece) {
                    out.push(id);
                    return;
                }
            }
        }
        if piece.len() <= CACHE_MAX_PIECE {
            if let Some(ids) = cache.map.get(piece) {
                out.extend_from_slice(ids);
                return;
            }
        }
        let start = out.len();
        if piece.len() <= 48 {
            self.bpe_small(piece, out);
        } else {
            self.bpe_large(piece, out);
        }
        if piece.len() <= CACHE_MAX_PIECE {
            if cache.map.len() >= CACHE_MAX {
                cache.map.clear();
            }
            cache.map.insert(piece.into(), out[start..].into());
        }
    }

    /// Quadratic BPE for short pieces: merge the lowest-ranked pair, leftmost first, until none is left.
    fn bpe_small(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut sym: [u32; 48] = [0; 48];
        let mut n = piece.len();
        for (i, &b) in piece.iter().enumerate() {
            sym[i] = self.byte_ids[b as usize];
        }
        loop {
            let mut best = u32::MAX;
            let mut best_i = 0;
            let mut best_id = 0;
            for i in 0..n - 1 {
                if let Some(&(rank, id)) = self.merges.get(&pair_key(sym[i], sym[i + 1])) {
                    if rank < best {
                        best = rank;
                        best_i = i;
                        best_id = id;
                    }
                }
            }
            if best == u32::MAX {
                break;
            }
            sym[best_i] = best_id;
            sym.copy_within(best_i + 2..n, best_i + 1);
            n -= 1;
            if n == 1 {
                break;
            }
        }
        out.extend_from_slice(&sym[..n]);
    }

    /// Heap-based BPE for long pieces, as `tokenizers`' Word::merge_all: min (rank, position) first.
    fn bpe_large(&self, piece: &[u8], out: &mut Vec<u32>) {
        let n = piece.len();
        let mut id: Vec<u32> = piece.iter().map(|&b| self.byte_ids[b as usize]).collect();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();
        let mut alive = vec![true; n];
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32)>> = BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(&(rank, nid)) = self.merges.get(&pair_key(id[i], id[i + 1])) {
                heap.push(Reverse((rank, i, nid)));
            }
        }
        while let Some(Reverse((rank, i, nid))) = heap.pop() {
            if !alive[i] || next[i] < 0 {
                continue;
            }
            let j = next[i] as usize;
            match self.merges.get(&pair_key(id[i], id[j])) {
                Some(&(r, m)) if r == rank && m == nid => {}
                _ => continue,
            }
            id[i] = nid;
            alive[j] = false;
            next[i] = next[j];
            if next[j] >= 0 {
                prev[next[j] as usize] = i as isize;
            }
            if prev[i] >= 0 {
                let p = prev[i] as usize;
                if let Some(&(r, m)) = self.merges.get(&pair_key(id[p], id[i])) {
                    heap.push(Reverse((r, p, m)));
                }
            }
            if next[i] >= 0 {
                let q = next[i] as usize;
                if let Some(&(r, m)) = self.merges.get(&pair_key(id[i], id[q])) {
                    heap.push(Reverse((r, i, m)));
                }
            }
        }
        let mut i = 0isize;
        while i >= 0 {
            out.push(id[i as usize]);
            i = next[i as usize];
        }
    }

    /// Byte spans of the added tokens that encoding finds in `text`, in order. Tokenisation restarts
    /// at each edge of one, so a text can be cut there and its parts encoded separately.
    pub fn added_spans(&self, text: &str) -> Vec<(usize, usize)> {
        let b = text.as_bytes();
        let spans = std::cell::RefCell::new(Vec::new());
        let mut gap = |a: usize, e: usize| self.added_norm.split_pos(b, a, e, &mut |_, _| {}, &mut |x, y| spans.borrow_mut().push((x, y)));
        self.added_raw.split_pos(b, 0, b.len(), &mut gap, &mut |x, y| spans.borrow_mut().push((x, y)));
        spans.into_inner()
    }

    /// Bytes of a token sequence (added tokens as their text).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut v = Vec::new();
        for &id in ids {
            v.extend_from_slice(self.token_bytes(id));
        }
        v
    }

    /// Text of a token sequence; invalid UTF-8 is replaced, as the reference decoder does.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// Incremental decoder: feed ids, get text as soon as the bytes form complete UTF-8 characters.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32, out: &mut String) {
        self.pending.extend_from_slice(tok.token_bytes(id));
        self.drain(out, false);
    }

    /// Flush whatever is left (an incomplete character becomes U+FFFD).
    pub fn finish(&mut self, out: &mut String) {
        self.drain(out, true);
    }

    fn drain(&mut self, out: &mut String, all: bool) {
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).unwrap());
                    match e.error_len() {
                        Some(bad) => {
                            // A byte sequence that can never become valid: replace it and go on.
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                        None => {
                            // Incomplete character at the end: wait for more bytes.
                            self.pending.drain(..good);
                            if all && !self.pending.is_empty() {
                                out.push('\u{FFFD}');
                                self.pending.clear();
                            }
                            return;
                        }
                    }
                }
            }
        }
    }
}
