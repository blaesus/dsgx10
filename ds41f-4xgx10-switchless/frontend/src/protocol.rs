//! Frames of the front-end ↔ rank-0 protocol (docs/protocol.md, engine/common/frontend_protocol.h).
//! An 8-byte header `{u32 size; u16 type; u16 reserved}`, then `size` payload bytes: a fixed
//! little-endian struct followed by arrays.

use tokio::io::{AsyncRead, AsyncReadExt};

pub const VERSION: u32 = 5;
pub const DEFAULT_SOCKET: &str = "/tmp/dsgb10-engine.sock";
/// Larger frames are a protocol error (a 1M-token prompt is 4 MB).
pub const MAX_FRAME: usize = 32 << 20; // 1M token IDs + <=4096 paths, never feature payloads

pub const T_HELLO: u16 = 0x01;
pub const T_SUBMIT: u16 = 0x02;
pub const T_CANCEL: u16 = 0x03;
pub const T_RELEASE: u16 = 0x04;
pub const T_HELLO_ENGINE: u16 = 0x81;
pub const T_TOKENS: u16 = 0x82;
pub const T_DONE: u16 = 0x83;
pub const T_IMAGE_MISS: u16 = 0x84;

pub const FLAG_SEED: u32 = 1;
pub const FLAG_KEEP: u32 = 2;
pub const FLAG_LOGPROBS: u32 = 4;
pub const FLAG_FULL_PREFIX: u32 = 16;
pub const FLAG_TAIL_PREFIX: u32 = 32;

pub const TOK_LOGPROBS: u32 = 1;
pub const TOK_TOP: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneReason {
    Stop = 0,
    Length = 1,
    Cancelled = 2,
    Context = 3,
    Error = 4,
}

impl DoneReason {
    pub fn from_u32(v: u32) -> DoneReason {
        match v {
            0 => DoneReason::Stop,
            1 => DoneReason::Length,
            2 => DoneReason::Cancelled,
            3 => DoneReason::Context,
            _ => DoneReason::Error,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageSpan {
    pub start: u32,
    pub rows: u32,
    pub digest: [u8; 32],
    pub feature: crate::feature_file::FeatureFile, // immutable file, little-endian BF16 [rows, 5120]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Submit {
    pub req_id: u64,
    pub session_id: u64,
    pub base_session_id: u64,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_new_tokens: u32,
    pub flags: u32,
    pub n_top_logprobs: u32,
    pub prefix_hint: u32,
    pub stop_ids: Vec<u32>,
    pub prompt: Vec<u32>,
    pub images: Vec<ImageSpan>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HelloEngine {
    pub version: u32,
    pub max_context: u32,
    pub vocab_size: u32,
    pub eos_id: u32,
    pub max_sessions: u32,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tokens {
    pub req_id: u64,
    pub ids: Vec<u32>,
    pub logprobs: Option<Vec<f32>>,
    /// n_top (id, logprob) pairs per token
    pub top: Option<Vec<(u32, f32)>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Done {
    pub req_id: u64,
    pub reason: DoneReason,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub queue_us: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub error: String,
}

/// A decoded frame, either direction.
#[derive(Clone, Debug, PartialEq)]
pub enum Frame {
    Hello { version: u32 },
    Submit(Submit),
    Cancel { req_id: u64 },
    Release { session_id: u64 },
    HelloEngine(HelloEngine),
    Tokens(Tokens),
    Done(Done),
    ImageMiss { req_id: u64, indices: Vec<u32> },
}

// ------------------------------------------------------------------ encoding

struct W(Vec<u8>);

impl W {
    fn new(ty: u16) -> W {
        let mut v = Vec::with_capacity(64);
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&ty.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        W(v)
    }
    fn u32(&mut self, x: u32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn u64(&mut self, x: u64) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn f32(&mut self, x: f32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }
    fn u32s(&mut self, xs: &[u32]) -> &mut Self {
        self.0.reserve(xs.len() * 4);
        for x in xs {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
        self
    }
    fn finish(mut self) -> Vec<u8> {
        let size = (self.0.len() - 8) as u32;
        self.0[..4].copy_from_slice(&size.to_le_bytes());
        self.0
    }
}

impl Frame {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Frame::Hello { version } => {
                let mut w = W::new(T_HELLO);
                w.u32(*version).u32(0);
                w.finish()
            }
            Frame::Submit(s) => {
                let mut w = W::new(T_SUBMIT);
                w.u64(s.req_id).u64(s.session_id).u64(s.base_session_id).u64(s.seed);
                w.f32(s.temperature).f32(s.top_p).u32(s.top_k).u32(s.max_new_tokens);
                w.u32(s.flags).u32(s.n_top_logprobs).u32(s.prefix_hint);
                w.u32(s.stop_ids.len() as u32).u32(s.prompt.len() as u32).u32(s.images.len() as u32);
                w.u32s(&s.stop_ids).u32s(&s.prompt);
                for image in &s.images {
                    w.u32(image.start).u32(image.rows);
                    w.0.extend_from_slice(&image.digest);
                    w.u32(image.feature.path().len() as u32);
                    w.0.extend_from_slice(image.feature.path().as_bytes());
                }
                w.finish()
            }
            Frame::ImageMiss { req_id, indices } => {
                let mut w = W::new(T_IMAGE_MISS);
                w.u64(*req_id).u32(indices.len() as u32).u32(0).u32s(indices);
                w.finish()
            }
            Frame::Cancel { req_id } => {
                let mut w = W::new(T_CANCEL);
                w.u64(*req_id);
                w.finish()
            }
            Frame::Release { session_id } => {
                let mut w = W::new(T_RELEASE);
                w.u64(*session_id);
                w.finish()
            }
            Frame::HelloEngine(h) => {
                let mut w = W::new(T_HELLO_ENGINE);
                w.u32(h.version).u32(h.max_context).u32(h.vocab_size).u32(h.eos_id).u32(h.max_sessions);
                w.u32(h.name.len() as u32);
                w.0.extend_from_slice(h.name.as_bytes());
                w.finish()
            }
            Frame::Tokens(t) => {
                let mut w = W::new(T_TOKENS);
                let flags = if t.logprobs.is_some() { TOK_LOGPROBS } else { 0 } | if t.top.is_some() { TOK_TOP } else { 0 };
                w.u64(t.req_id).u32(t.ids.len() as u32).u32(flags).u32s(&t.ids);
                if let Some(lp) = &t.logprobs {
                    for x in lp {
                        w.f32(*x);
                    }
                }
                if let Some(top) = &t.top {
                    for (id, lp) in top {
                        w.u32(*id).f32(*lp);
                    }
                }
                w.finish()
            }
            Frame::Done(d) => {
                let mut w = W::new(T_DONE);
                w.u64(d.req_id).u32(d.reason as u32).u32(d.prompt_tokens).u32(d.cached_tokens).u32(d.completion_tokens);
                w.u64(d.queue_us).u64(d.prefill_us).u64(d.decode_us).u32(d.error.len() as u32).u32(0);
                w.0.extend_from_slice(d.error.as_bytes());
                w.finish()
            }
        }
    }

    /// Decode one frame's payload. `n_top` is the request's n_top_logprobs, needed to size the
    /// top-logprob array of a TOKENS frame (pass 0 if it is not known; such arrays are then rejected).
    pub fn decode(ty: u16, p: &[u8], n_top: impl Fn(u64) -> u32) -> Result<Frame, String> {
        let mut r = R { p, i: 0 };
        let f = match ty {
            T_HELLO => {
                let version = r.u32()?;
                r.u32()?;
                Frame::Hello { version }
            }
            T_SUBMIT => {
                let req_id = r.u64()?;
                let session_id = r.u64()?;
                let base_session_id = r.u64()?;
                let seed = r.u64()?;
                let temperature = r.f32()?;
                let top_p = r.f32()?;
                let top_k = r.u32()?;
                let max_new_tokens = r.u32()?;
                let flags = r.u32()?;
                let n_top_logprobs = r.u32()?;
                let prefix_hint = r.u32()?;
                let n_stop = r.u32()? as usize;
                let n_prompt = r.u32()? as usize;
                let n_images = r.u32()? as usize;
                if n_images > crate::feature_file::MAX_IMAGES || n_images > n_prompt { return Err("more image spans than prompt tokens".into()); }
                let stop_ids = r.u32s(n_stop)?;
                let prompt = r.u32s(n_prompt)?;
                let mut images = Vec::new();
                let mut end = 0u64;
                let mut image_rows = 0usize;
                for _ in 0..n_images {
                    let start = r.u32()?;
                    let rows = r.u32()?;
                    if rows == 0 || rows > 1024 || (start as u64) < end || start as u64 + rows as u64 > n_prompt as u64 {
                        return Err("invalid image span".into());
                    }
                    end = start as u64 + rows as u64;
                    image_rows = image_rows.checked_add(rows as usize).ok_or("image rows overflow")?;
                    if image_rows > crate::feature_file::MAX_IMAGE_ROWS { return Err("image rows exceed context budget".into()); }
                    let digest = r.bytes(32)?.try_into().unwrap();
                    let size = r.u32()? as usize;
                    if size == 0 || size > 4096 { return Err("invalid feature path length".into()); }
                    let path = std::str::from_utf8(r.bytes(size)?).map_err(|_| "invalid feature path UTF8")?;
                    let feature = crate::feature_file::FeatureFile::descriptor(path, rows as usize * 5120 * 2)?;
                    images.push(ImageSpan { start, rows, digest, feature });
                }
                Frame::Submit(Submit {
                    req_id,
                    session_id,
                    base_session_id,
                    seed,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    flags,
                    n_top_logprobs,
                    prefix_hint,
                    stop_ids,
                    prompt,
                    images,
                })
            }
            T_IMAGE_MISS => {
                let req_id = r.u64()?; let n = r.u32()? as usize;
                if r.u32()? != 0 || n == 0 || n > 52428 { return Err("bad image miss frame".into()); }
                Frame::ImageMiss { req_id, indices: r.u32s(n)? }
            }
            T_CANCEL => Frame::Cancel { req_id: r.u64()? },
            T_RELEASE => Frame::Release { session_id: r.u64()? },
            T_HELLO_ENGINE => {
                let version = r.u32()?;
                let max_context = r.u32()?;
                let vocab_size = r.u32()?;
                let eos_id = r.u32()?;
                let max_sessions = r.u32()?;
                let n = r.u32()? as usize;
                let name = String::from_utf8_lossy(r.bytes(n)?).into_owned();
                Frame::HelloEngine(HelloEngine { version, max_context, vocab_size, eos_id, max_sessions, name })
            }
            T_TOKENS => {
                let req_id = r.u64()?;
                let n = r.u32()? as usize;
                let flags = r.u32()?;
                let ids = r.u32s(n)?;
                let logprobs = if flags & TOK_LOGPROBS != 0 { Some((0..n).map(|_| r.f32()).collect::<Result<Vec<_>, _>>()?) } else { None };
                let top = if flags & TOK_TOP != 0 {
                    let k = n_top(req_id) as usize;
                    Some((0..n * k).map(|_| Ok((r.u32()?, r.f32()?))).collect::<Result<Vec<_>, String>>()?)
                } else {
                    None
                };
                Frame::Tokens(Tokens { req_id, ids, logprobs, top })
            }
            T_DONE => {
                let req_id = r.u64()?;
                let reason = DoneReason::from_u32(r.u32()?);
                let prompt_tokens = r.u32()?;
                let cached_tokens = r.u32()?;
                let completion_tokens = r.u32()?;
                let queue_us = r.u64()?;
                let prefill_us = r.u64()?;
                let decode_us = r.u64()?;
                let n = r.u32()? as usize;
                r.u32()?;
                let error = String::from_utf8_lossy(r.bytes(n)?).into_owned();
                Frame::Done(Done { req_id, reason, prompt_tokens, cached_tokens, completion_tokens, queue_us, prefill_us, decode_us, error })
            }
            _ => return Err(format!("unknown frame type {ty:#x}")),
        };
        if r.i != p.len() {
            return Err(format!("frame type {ty:#x}: {} trailing bytes", p.len() - r.i));
        }
        Ok(f)
    }
}

struct R<'a> {
    p: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p.len() - self.i < n {
            return Err("truncated frame".into());
        }
        let b = &self.p[self.i..self.i + n];
        self.i += n;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let b = self.bytes(n.checked_mul(4).ok_or("bad length")?)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Read one frame: (type, payload). Ok(None) on a clean end of stream between frames.
pub async fn read_frame<Rd: AsyncRead + Unpin>(rd: &mut Rd) -> std::io::Result<Option<(u16, Vec<u8>)>> {
    let mut h = [0u8; 8];
    match rd.read_exact(&mut h).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let size = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
    let ty = u16::from_le_bytes(h[4..6].try_into().unwrap());
    if size > MAX_FRAME {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("frame of {size} bytes")));
    }
    let mut p = vec![0u8; size];
    rd.read_exact(&mut p).await?;
    Ok(Some((ty, p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_resolution_history_exceeding_512_mib_is_metadata_only() {
        let mut s=Submit { req_id:1,session_id:1,base_session_id:0,seed:1017,temperature:0.,top_p:1.,
            top_k:0,max_new_tokens:1,flags:FLAG_KEEP,n_top_logprobs:0,prefix_hint:0,
            stop_ids:vec![1],prompt:vec![129264;58*1024],images:Vec::new() };
        s.images=(0..58).map(|i|ImageSpan { start:i*1024,rows:1024,digest:[i as u8;32],
            feature:crate::feature_file::FeatureFile::descriptor(
                &format!("/tmp/dsgb10-vision-features/test/f{i}"),1024*5120*2).unwrap() }).collect();
        assert_eq!(s.images.iter().map(|i|i.feature.len()).sum::<usize>(),580<<20);
        let b=Frame::Submit(s.clone()).encode();
        assert!(b.len()<256<<10,"all historical feature bytes must stay outside the frame");
        assert_eq!(Frame::decode(T_SUBMIT,&b[8..],|_|0).unwrap(),Frame::Submit(s));
        // Bad root / path truncation / trailing inline bytes are protocol errors.
        let mut bad=b.clone();
        let path_at=8+72+4+58*1024*4+44;
        bad[path_at]=b'x';
        assert!(Frame::decode(T_SUBMIT,&bad[8..],|_|0).is_err());
        assert!(Frame::decode(T_SUBMIT,&b[8..b.len()-1],|_|0).is_err());
        let mut bad=b.clone();bad.push(0);
        assert!(Frame::decode(T_SUBMIT,&bad[8..],|_|0).is_err());
        let mut bad=b;bad[8+68..8+72].copy_from_slice(&4097u32.to_le_bytes());
        assert!(Frame::decode(T_SUBMIT,&bad[8..],|_|0).is_err());
    }

    #[test]
    fn roundtrip_and_layout() {
        let s = Submit {
            req_id: 7,
            session_id: 9,
            base_session_id: 3,
            seed: 42,
            temperature: 0.5,
            top_p: 0.95,
            top_k: 20,
            max_new_tokens: 100,
            flags: FLAG_SEED | FLAG_KEEP,
            n_top_logprobs: 0,
            prefix_hint: 5,
            stop_ids: vec![1],
            prompt: vec![0, 10, 20],
            images: Vec::new(),
        };
        let b = Frame::Submit(s.clone()).encode();
        assert_eq!(b.len(), 8 + 72 + 4 + 12); // header, SubmitHeader, stop id, prompt
        assert_eq!(u32::from_le_bytes(b[8 + 64..8 + 68].try_into().unwrap()), 3); // n_prompt at offset 64
        assert_eq!(Frame::decode(T_SUBMIT, &b[8..], |_| 0).unwrap(), Frame::Submit(s));
        let mut image_submit = match Frame::decode(T_SUBMIT, &b[8..], |_| 0).unwrap() {
            Frame::Submit(s) => s, _ => unreachable!()
        };
        image_submit.prompt=vec![0,129264,129264,10];
        image_submit.images=vec![ImageSpan { start:1, rows:2, digest:[7;32], feature:crate::feature_file::FeatureFile::descriptor("/tmp/dsgb10-vision-features/test/file",2*5120*2).unwrap() }];
        let image_frame=Frame::Submit(image_submit.clone()).encode();
        assert_eq!(Frame::decode(T_SUBMIT,&image_frame[8..],|_|0).unwrap(),Frame::Submit(image_submit.clone()));
        assert!(Frame::decode(T_SUBMIT,&image_frame[8..image_frame.len()-1],|_|0).is_err());
        // An agent can accumulate many historical images; preserve every span.
        let mut history=image_submit.clone();
        history.prompt=vec![129264;128];
        history.images=(0..64).map(|i| ImageSpan {start:i*2, rows:2, digest:[i as u8;32], feature:crate::feature_file::FeatureFile::descriptor("/tmp/dsgb10-vision-features/test/file",2*5120*2).unwrap()}).collect();
        let history_frame=Frame::Submit(history.clone()).encode();
        assert_eq!(Frame::decode(T_SUBMIT,&history_frame[8..],|_|0).unwrap(),Frame::Submit(history.clone()));
        history.images[63].start=125; // overlaps the preceding image
        assert!(Frame::decode(T_SUBMIT,&Frame::Submit(history).encode()[8..],|_|0).is_err());
        image_submit.images[0].start=4;
        assert!(Frame::decode(T_SUBMIT,&Frame::Submit(image_submit).encode()[8..],|_|0).is_err());
        let d = Done {
            req_id: 1,
            reason: DoneReason::Length,
            prompt_tokens: 3,
            cached_tokens: 2,
            completion_tokens: 4,
            queue_us: 5,
            prefill_us: 6,
            decode_us: 7,
            error: "x".into(),
        };
        let b = Frame::Done(d.clone()).encode();
        assert_eq!(b.len(), 8 + 56 + 1);
        assert_eq!(Frame::decode(T_DONE, &b[8..], |_| 0).unwrap(), Frame::Done(d));
        let t = Tokens { req_id: 2, ids: vec![5, 6], logprobs: Some(vec![-0.5, -1.0]), top: Some(vec![(5, -0.5), (6, -1.0)]) };
        let b = Frame::Tokens(t.clone()).encode();
        assert_eq!(Frame::decode(T_TOKENS, &b[8..], |_| 1).unwrap(), Frame::Tokens(t));
    }
}
