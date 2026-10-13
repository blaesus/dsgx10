//! A rank-0 stand-in for testing the front-end end to end: speaks the protocol, keeps sessions
//! (reporting cached_tokens from the longest common prefix), and streams scripted text.
//!
//! Script, from the decoded prompt: reasoning if the prompt ends in `<think>`; then, for the last
//! user turn, `MOCK_TOOL <name> <json args>` → that tool call (if tools are in the prompt),
//! `MOCK_LOOP` → a repeating sentence, `MOCK_LONG <n>` → n numbered filler words, a tool result → a reply
//! quoting it, anything else → a reply quoting the user.

use crate::encoding::{render_tool_calls, ASSISTANT_SP, THINK_END, THINK_START, USER_SP};
use crate::protocol::{read_frame, Done, DoneReason, Frame, HelloEngine, Submit, Tokens, FLAG_KEEP, FLAG_LOGPROBS, VERSION};
use crate::tokenizer::Tokenizer;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct MockConfig {
    pub socket: PathBuf,
    pub tok: Arc<Tokenizer>,
    /// per generated token
    pub delay: Duration,
    /// simulated prefill speed for uncached prompt tokens
    pub prefill_tok_per_s: f64,
    pub max_context: u32,
}

type Sessions = Arc<Mutex<HashMap<u64, Vec<u32>>>>;

pub async fn run(cfg: MockConfig) -> std::io::Result<()> {
    let _ = std::fs::remove_file(&cfg.socket);
    let listener = UnixListener::bind(&cfg.socket)?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&cfg.socket, std::fs::Permissions::from_mode(0o600))?;
    }
    eprintln!("mock engine: listening on {}", cfg.socket.display());
    let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
    loop {
        let (stream, _) = listener.accept().await?;
        eprintln!("mock engine: front-end connected");
        tokio::spawn(connection(stream, cfg.clone(), sessions.clone()));
    }
}

async fn connection(stream: UnixStream, cfg: MockConfig, sessions: Sessions) {
    let (mut rd, mut wr) = stream.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    tokio::spawn(async move {
        while let Some(b) = rx.recv().await {
            if wr.write_all(&b).await.is_err() {
                break;
            }
        }
    });
    let hello = HelloEngine {
        version: VERSION,
        max_context: cfg.max_context,
        vocab_size: cfg.tok.vocab_size() as u32,
        eos_id: 1,
        max_sessions: 64,
        name: "mock".into(),
    };
    let _ = tx.send(Frame::HelloEngine(hello).encode());
    let cancels: Arc<Mutex<HashMap<u64, Arc<AtomicBool>>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut greeted = false;
    loop {
        let (ty, p) = match read_frame(&mut rd).await {
            Ok(Some(f)) => f,
            _ => break,
        };
        let frame = match Frame::decode(ty, &p, |_| 0) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("mock engine: bad frame: {e}");
                break;
            }
        };
        match frame {
            Frame::Hello { version } => {
                if version != VERSION {
                    eprintln!("mock engine: front-end speaks version {version}");
                    break;
                }
                greeted = true;
            }
            _ if !greeted => {
                eprintln!("mock engine: frame before HELLO");
                break;
            }
            Frame::Submit(s) => {
                let flag = Arc::new(AtomicBool::new(false));
                cancels.lock().unwrap().insert(s.req_id, flag.clone());
                let (tx, cfg, sessions, cancels) = (tx.clone(), cfg.clone(), sessions.clone(), cancels.clone());
                tokio::spawn(async move {
                    let req_id = s.req_id;
                    generate(s, flag, &tx, &cfg, &sessions).await;
                    cancels.lock().unwrap().remove(&req_id);
                });
            }
            Frame::Cancel { req_id } => {
                if let Some(f) = cancels.lock().unwrap().get(&req_id) {
                    f.store(true, Ordering::Relaxed);
                }
            }
            Frame::Release { session_id } => {
                sessions.lock().unwrap().remove(&session_id);
            }
            f => {
                eprintln!("mock engine: unexpected frame {f:?}");
                break;
            }
        }
    }
    for f in cancels.lock().unwrap().values() {
        f.store(true, Ordering::Relaxed);
    }
    eprintln!("mock engine: front-end disconnected");
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

async fn generate(s: Submit, cancel: Arc<AtomicBool>, tx: &mpsc::UnboundedSender<Vec<u8>>, cfg: &MockConfig, sessions: &Sessions) {
    let t0 = Instant::now();
    let mut done = Done {
        req_id: s.req_id,
        reason: DoneReason::Error,
        prompt_tokens: s.prompt.len() as u32,
        cached_tokens: 0,
        completion_tokens: 0,
        queue_us: 0,
        prefill_us: 0,
        decode_us: 0,
        error: String::new(),
    };
    if s.prompt.is_empty() || s.prompt.len() >= cfg.max_context as usize || s.max_new_tokens == 0 || s.session_id == 0 {
        done.error = format!("bad request: {} prompt tokens, max_new_tokens {}, session {}", s.prompt.len(), s.max_new_tokens, s.session_id);
        let _ = tx.send(Frame::Done(done).encode());
        return;
    }
    let cached = {
        let ss = sessions.lock().unwrap();
        let hist = ss.get(&s.session_id).or_else(|| if s.base_session_id != 0 { ss.get(&s.base_session_id) } else { None });
        hist.map_or(0, |h| lcp(h, &s.prompt)).min(s.prompt.len() - 1)
    };
    done.cached_tokens = cached as u32;
    let todo = s.prompt.len() - cached;
    if cfg.prefill_tok_per_s > 0.0 {
        tokio::time::sleep(Duration::from_secs_f64(todo as f64 / cfg.prefill_tok_per_s)).await;
    }
    let t_first = Instant::now();
    done.prefill_us = (t_first - t0).as_micros() as u64;

    let prompt_text = cfg.tok.decode(&s.prompt);
    let script = script(&prompt_text);
    let pre = cfg.tok.encode(&script.prefix);
    let mut body = cfg.tok.encode(&script.body);
    if !script.endless {
        body.push(1);
    }
    let mut generated: Vec<u32> = Vec::new();
    let mut i = 0;
    done.reason = loop {
        if cancel.load(Ordering::Relaxed) {
            break DoneReason::Cancelled;
        }
        if generated.len() >= s.max_new_tokens as usize {
            break DoneReason::Length;
        }
        if s.prompt.len() + generated.len() >= cfg.max_context as usize {
            break DoneReason::Context;
        }
        let id = if i < pre.len() { pre[i] } else { body[(i - pre.len()) % body.len()] };
        i += 1;
        generated.push(id);
        let logprobs = (s.flags & FLAG_LOGPROBS != 0).then(|| vec![0.0f32]);
        let _ = tx.send(Frame::Tokens(Tokens { req_id: s.req_id, ids: vec![id], logprobs, top: None }).encode());
        if s.stop_ids.contains(&id) {
            break DoneReason::Stop;
        }
        if !script.endless && i >= pre.len() + body.len() {
            break DoneReason::Length; // the script ran out without a stop id (ignore_eos)
        }
        if !cfg.delay.is_zero() {
            tokio::time::sleep(cfg.delay).await;
        }
    };
    done.completion_tokens = generated.len() as u32;
    done.decode_us = t_first.elapsed().as_micros() as u64;
    if s.flags & FLAG_KEEP != 0 {
        // the last sampled id has no KV yet
        let mut hist = s.prompt.clone();
        hist.extend_from_slice(&generated[..generated.len().saturating_sub(1)]);
        sessions.lock().unwrap().insert(s.session_id, hist);
    } else {
        sessions.lock().unwrap().remove(&s.session_id);
    }
    let _ = tx.send(Frame::Done(done).encode());
}

struct Script {
    /// reasoning (and `</think>`), sent once
    prefix: String,
    /// the answer, then EOS; repeated without EOS if endless
    body: String,
    endless: bool,
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.trim().chars().take(n).collect();
    t.replace('\n', " ")
}

fn script(prompt: &str) -> Script {
    let thinking = prompt.ends_with(THINK_START);
    let has_tools = prompt.contains("## Tools\n\nYou have access to a set of tools");
    let last_user = prompt.rfind(USER_SP).map_or("", |p| {
        let seg = &prompt[p + USER_SP.len()..];
        seg.find(ASSISTANT_SP).map_or(seg, |e| &seg[..e])
    });
    let tool_result = last_user.find("<tool_result>").map(|a| {
        let r = &last_user[a + "<tool_result>".len()..];
        r.find("</tool_result>").map_or(r, |e| &r[..e])
    });
    let mut prefix = String::new();
    if thinking {
        prefix.push_str(&format!("Mock reasoning: the last turn has {} characters.", last_user.chars().count()));
        prefix.push_str(THINK_END);
    }
    let mut text = String::new();
    let mut endless = false;
    if let Some(r) = tool_result {
        text.push_str(&format!("The tool returned: {}", snippet(r, 200)));
    } else if let (Some(p), true) = (last_user.find("MOCK_TOOL "), has_tools) {
        let rest = &last_user[p + "MOCK_TOOL ".len()..];
        let (name, args) = rest.split_once(' ').unwrap_or((rest.trim(), "{}"));
        let args: Value = crate::pyjson::loads(args.trim()).unwrap_or(Value::Object(Default::default()));
        text.push_str("Calling the tool.");
        text.push_str(&render_tool_calls(&[(Value::String(name.trim().to_string()), args)]));
    } else if last_user.contains("MOCK_LOOP") {
        text.push_str("Let me think about this again. ");
        endless = true;
    } else if let Some(p) = last_user.find("MOCK_LONG") {
        let n: usize = last_user[p + 9..].split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(1000);
        for k in 0..n {
            text.push_str(&format!("word{k} ")); // not periodic, so the loop guard leaves it alone
        }
    } else {
        text.push_str(&format!("Mock reply to: {}", snippet(last_user, 200)));
    }
    Script { prefix, body: text, endless }
}
