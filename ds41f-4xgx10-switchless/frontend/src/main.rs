use dsfront::tokenizer::Tokenizer;
use std::io::BufRead;
use std::time::Instant;

const DEFAULT_TOKENIZER: &str = "/data/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json";

fn usage() -> ! {
    eprintln!(
        "usage:
  dsfront serve [options]                             the OpenAI-compatible server
      --listen ADDR             127.0.0.1:18890; repeat it for more (loopback or specific addresses;
                                never 0.0.0.0 or ::)
      --engine-socket PATH      /tmp/dsgb10-engine.sock
      --api-key-file PATH       require this bearer key (the file's content, trimmed)
      --model-name NAME         deepseek-v4.1-flash-uncensored
      --max-tokens N            default and cap for max_tokens: 32768
      --effort N                reasoning effort budget when a request names none: 75
      --thinking                thinking on when a request does not say (default off)
      --temperature T --top-p P sampling defaults: 1.0 and 0.95
      --no-loop-guard           do not stop repetition loops
      --dump-dir DIR            write each request, prompt and ids to DIR, and at its end the output's
                                ids (N.out.json; debugging and bench traces)
      --hold-s N                while the engine is down (the ring lent, or a restart), a request
                                waits this long (600), then is answered 503 with Retry-After: 60
      --admin-socket PATH       commands from service.sh (mode 600): status, hold IDLE_S, release
      --table-file PATH         the session table, saved here at a stop and taken back at the next
                                start (mode 600, in a directory of mode 700; it holds the
                                conversations' text), so that the engine's kept sessions are found again
      --table-days N            sessions unused for longer are not kept: 7
  dsfront admin --socket PATH <command>               one command to a server's admin socket; prints
                                                      the answer (docs/engine.md, \"Launch and stop\")
  dsfront mock-engine [options]                       a rank-0 stand-in (docs/protocol.md)
      --socket PATH             /tmp/dsgb10-engine.sock
      --delay-ms N              per generated token: 5
      --prefill-tps N           simulated prefill tokens/s: 20000
      --max-context N           1048576
  dsfront tokcheck <cases.jsonl>                      compare against reference ids ({{\"text\",\"ids\"}} per line)
  dsfront enccheck <cases.jsonl>                      compare the chat encoding with encoding.py's (tools/gen_encoding_cases.py)
  dsfront tokenize <file>                             print ids of a file's text
  dsfront tokbench <file>                             tokenisation throughput
  every command that tokenises takes --tokenizer PATH ({DEFAULT_TOKENIZER})"
    );
    std::process::exit(2)
}

fn parse_opt<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match opt(args, name) {
        None => default,
        Some(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("bad value for {name}: {v}");
            std::process::exit(2)
        }),
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime")
}

fn serve(args: &[String]) {
    use dsfront::server::{router, App, ServerConfig};
    let mut listen: Vec<std::net::SocketAddr> = opts(args, "--listen")
        .iter()
        .map(|v| {
            v.parse().unwrap_or_else(|_| {
                eprintln!("bad value for --listen: {v}");
                std::process::exit(2)
            })
        })
        .collect();
    if listen.is_empty() {
        listen.push("127.0.0.1:18890".parse().unwrap());
    }
    for a in &listen {
        if unspecified(a.ip()) {
            eprintln!("refusing to listen on {a}: bind to loopback or specific addresses");
            std::process::exit(2);
        }
    }
    let api_key = opt(args, "--api-key-file").map(|p| {
        let k = std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("{p}: {e}");
            std::process::exit(1)
        });
        let k = k.trim().to_string();
        if k.is_empty() {
            eprintln!("{p} is empty");
            std::process::exit(1);
        }
        k
    });
    if api_key.is_none() {
        eprintln!("warning: no --api-key-file, requests are not authenticated");
    }
    let cfg = ServerConfig {
        model_name: opt(args, "--model-name").unwrap_or_else(|| "deepseek-v4.1-flash-uncensored".into()),
        api_key,
        max_tokens_cap: parse_opt(args, "--max-tokens", 32768u32),
        default_effort: parse_opt(args, "--effort", 75i64).clamp(1, 100),
        default_thinking: args.iter().any(|a| a == "--thinking"),
        default_temperature: parse_opt(args, "--temperature", 1.0f32),
        default_top_p: parse_opt(args, "--top-p", 0.95f32),
        loop_guard: if args.iter().any(|a| a == "--no-loop-guard") { None } else { Some(Default::default()) },
        tokenize_threads: parse_opt(args, "--tokenize-threads", 8usize).max(1),
        dump_dir: opt(args, "--dump-dir").map(Into::into),
        hold: std::time::Duration::from_secs(parse_opt(args, "--hold-s", 600u64)),
    };
    let admin = opt(args, "--admin-socket");
    let table = opt(args, "--table-file").map(std::path::PathBuf::from);
    let table_age_s = (parse_opt(args, "--table-days", 7f64) * 86400.0) as u64;
    let tok = std::sync::Arc::new(load_tokenizer(args));
    let socket = opt(args, "--engine-socket").unwrap_or_else(|| dsfront::protocol::DEFAULT_SOCKET.into());
    runtime().block_on(async move {
        let engine = dsfront::engine::EngineClient::start(socket.into());
        let app = App::new(cfg, tok, engine);
        if let Some(t) = &table {
            eprintln!("{}", app.load_sessions(t, table_age_s));
        }
        let saver = app.clone();
        if let Some(path) = admin {
            let (gate, engine) = (app.gate.clone(), app.engine.clone());
            tokio::spawn(async move {
                if let Err(e) = dsfront::gate::admin(gate, engine, path.clone().into()).await {
                    eprintln!("admin socket {path}: {e}");
                    std::process::exit(1);
                }
            });
        }
        // every address bound before any is served: one that cannot be bound stops the start
        let mut listeners = Vec::new();
        for a in &listen {
            listeners.push(tokio::net::TcpListener::bind(a).await.unwrap_or_else(|e| {
                eprintln!("cannot listen on {a}: {e}");
                std::process::exit(1)
            }));
        }
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let gate = app.gate.clone();
        tokio::spawn(async move {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!("shutting down");
            gate.close(); // requests waiting at the gate are answered now, not at the hold limit
            let _ = stop_tx.send(true);
        });
        let routes = router(app);
        let mut servers = tokio::task::JoinSet::new();
        for (a, l) in listen.iter().zip(listeners) {
            eprintln!("serving on http://{a}/v1");
            let mut stop = stop_rx.clone();
            let shutdown = async move {
                let _ = stop.wait_for(|s| *s).await;
            };
            servers.spawn(std::future::IntoFuture::into_future(axum::serve(l, routes.clone()).with_graceful_shutdown(shutdown)));
        }
        while let Some(r) = servers.join_next().await {
            r.expect("server task").expect("server");
        }
        if let Some(t) = &table {
            eprintln!("{}", saver.save_sessions(t, table_age_s));
        }
    });
}

/// 0.0.0.0, ::, and :: with 0.0.0.0 mapped into it: every address of the box
fn unspecified(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_unspecified(),
        std::net::IpAddr::V6(v6) => v6.is_unspecified() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_unspecified()),
    }
}

/// One command to a server's admin socket (gate.rs); prints the one-line answer. Exit 2 when there
/// is no answer.
fn admin(args: &[String]) {
    use std::io::{Read, Write};
    let sock = opt(args, "--socket").unwrap_or_else(|| usage());
    let mut words = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--socket" {
            i += 2;
            continue;
        }
        words.push(args[i].as_str());
        i += 1;
    }
    if words.is_empty() {
        usage();
    }
    let fail = |e: std::io::Error| -> ! {
        eprintln!("{sock}: {e}");
        std::process::exit(2)
    };
    let mut s = std::os::unix::net::UnixStream::connect(&sock).unwrap_or_else(|e| fail(e));
    s.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap_or_else(|e| fail(e));
    writeln!(s, "{}", words.join(" ")).unwrap_or_else(|e| fail(e));
    let mut reply = String::new();
    s.read_to_string(&mut reply).unwrap_or_else(|e| fail(e));
    if reply.trim().is_empty() {
        eprintln!("{sock}: no answer");
        std::process::exit(2);
    }
    print!("{reply}");
}

fn mock_engine(args: &[String]) {
    let cfg = dsfront::mock::MockConfig {
        socket: opt(args, "--socket").unwrap_or_else(|| dsfront::protocol::DEFAULT_SOCKET.into()).into(),
        tok: std::sync::Arc::new(load_tokenizer(args)),
        delay: std::time::Duration::from_millis(parse_opt(args, "--delay-ms", 5u64)),
        prefill_tok_per_s: parse_opt(args, "--prefill-tps", 20000f64),
        max_context: parse_opt(args, "--max-context", 1048576u32),
    };
    if let Err(e) = runtime().block_on(dsfront::mock::run(cfg)) {
        eprintln!("mock engine: {e}");
        std::process::exit(1);
    }
}

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Every value of a repeatable option.
fn opts(args: &[String], name: &str) -> Vec<String> {
    args.iter().enumerate().filter(|(_, a)| *a == name).filter_map(|(i, _)| args.get(i + 1).cloned()).collect()
}

fn load_tokenizer(args: &[String]) -> Tokenizer {
    let path = opt(args, "--tokenizer").unwrap_or_else(|| DEFAULT_TOKENIZER.to_string());
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(&path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    eprintln!("tokenizer loaded in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    tok
}

fn tokcheck(args: &[String]) {
    let path = args.get(2).unwrap_or_else(|| usage());
    let tok = load_tokenizer(args);
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let (mut n, mut bad, mut bad_decode, mut bytes, mut toks) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut secs = 0f64;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("bad json line");
        let text = v["text"].as_str().expect("text");
        let want: Vec<u32> = v["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let t0 = Instant::now();
        let got = tok.encode(text);
        secs += t0.elapsed().as_secs_f64();
        n += 1;
        bytes += text.len();
        toks += want.len();
        if got != want {
            bad += 1;
            if bad <= 10 {
                let k = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                let lo = k.saturating_sub(3);
                eprintln!(
                    "MISMATCH case {n} ({} bytes): first diff at token {k}\n  text: {:?}\n  want[{lo}..]: {:?}\n  got [{lo}..]: {:?}",
                    text.len(),
                    text.chars().take(200).collect::<String>(),
                    &want[lo..(k + 4).min(want.len())],
                    &got[lo..(k + 4).min(got.len())]
                );
            }
        }
        if tok.decode(&want) != text {
            bad_decode += 1;
            if bad_decode <= 5 {
                eprintln!("DECODE MISMATCH case {n}: {:?}", text.chars().take(120).collect::<String>());
            }
        }
    }
    println!(
        "{n} cases, {bad} encode mismatches, {bad_decode} decode mismatches; {toks} tokens, {:.1} MB in {:.2} s = {:.2} M tok/s",
        bytes as f64 / 1e6,
        secs,
        toks as f64 / secs / 1e6
    );
    if bad > 0 || bad_decode > 0 {
        std::process::exit(1);
    }
}

/// Compare encoding.rs with encoding.py's output (tools/gen_encoding_cases.py).
fn enccheck(args: &[String]) {
    use dsfront::encoding::{encode_messages, parse_message_from_completion_text, Delta, EncodeOptions, StreamParser, ThinkingMode};
    use serde_json::Value;
    let path = args.get(2).unwrap_or_else(|| usage());
    let f = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    let (mut n_enc, mut n_parse, mut n_stream, mut bad) = (0usize, 0usize, 0usize, 0usize);
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut report = |name: &str, what: String| {
        bad += 1;
        if bad <= 12 {
            eprintln!("MISMATCH {name}: {what}");
        }
    };
    let t0 = Instant::now();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.unwrap();
        let c: Value = serde_json::from_str(&line).expect("bad json line");
        let name = c["name"].as_str().unwrap_or("?").to_string();
        let mode = ThinkingMode::parse(c["thinking_mode"].as_str().unwrap()).unwrap();
        match c["kind"].as_str() {
            Some("encode") => {
                n_enc += 1;
                let msgs = c["messages"].as_array().cloned().unwrap_or_default();
                let ctx = c["context"].as_array().cloned();
                let mut o = EncodeOptions::new(mode);
                o.context = ctx.as_deref();
                o.drop_thinking = c["drop_thinking"].as_bool().unwrap();
                o.add_default_bos_token = c["add_bos"].as_bool().unwrap();
                o.reasoning_effort = c.get("reasoning_effort");
                let got = encode_messages(&msgs, &o);
                match (got, c.get("prompt")) {
                    (Ok((p, imgs)), Some(Value::String(want))) => {
                        if &p != want {
                            let k = p.chars().zip(want.chars()).position(|(a, b)| a != b).unwrap_or(p.len().min(want.len()));
                            let at = |s: &str| s.chars().skip(k.saturating_sub(20)).take(60).collect::<String>();
                            report(&name, format!("prompt differs at char {k}:\n  want {:?}\n  got  {:?}", at(want), at(&p)));
                        } else if Value::Array(imgs) != c["images"] {
                            report(&name, "image records differ".into());
                        }
                    }
                    (Err(e), Some(Value::String(_))) => report(&name, format!("port raised {e:?}, encoding.py did not")),
                    (Ok((p, _)), None) => {
                        report(&name, format!("encoding.py raised {}, port gave {:?}", c["error"], p.chars().take(80).collect::<String>()))
                    }
                    _ => {}
                }
            }
            Some("parse") => {
                n_parse += 1;
                let text = c["text"].as_str().unwrap();
                let got = parse_message_from_completion_text(text, mode);
                let want = c.get("result");
                match (&got, want) {
                    (Ok(m), Some(w)) => {
                        let calls: Vec<(String, String)> = w["tool_calls"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|t| (t["function"]["name"].as_str().unwrap().to_string(), t["function"]["arguments"].as_str().unwrap().to_string()))
                            .collect();
                        if m.content != w["content"].as_str().unwrap()
                            || m.reasoning_content != w["reasoning_content"].as_str().unwrap()
                            || m.tool_calls != calls
                        {
                            report(&name, format!("parse differs: got {m:?}\n  want {w}"));
                        }
                    }
                    (Err(e), Some(_)) => report(&name, format!("port raised {e:?}, encoding.py did not")),
                    (Ok(m), None) => report(&name, format!("encoding.py raised {}, port gave {m:?}", c["error"])),
                    _ => {}
                }
                // The streaming parser, fed in random pieces, must agree wherever the reference parses.
                let chars: Vec<char> = text.chars().collect();
                let mut p = StreamParser::new(mode);
                let mut out = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let k = 1 + (rng % 9) as usize;
                    let piece: String = chars[i..(i + k).min(chars.len())].iter().collect();
                    p.push(&piece, &mut out);
                    i += k;
                }
                p.finish(&mut out);
                if let Ok(m) = got {
                    n_stream += 1;
                    let (mut r, mut ct, mut calls) = (String::new(), String::new(), Vec::new());
                    for d in out {
                        match d {
                            Delta::Reasoning(s) => r.push_str(&s),
                            Delta::Content(s) => ct.push_str(&s),
                            Delta::ToolCall(_, n, a) => calls.push((n, a)),
                        }
                    }
                    if r != m.reasoning_content || ct != m.content || calls != m.tool_calls {
                        report(&name, format!("stream differs: reasoning {r:?} content {ct:?} calls {calls:?}\n  want {m:?}"));
                    }
                }
            }
            _ => panic!("unknown case kind"),
        }
    }
    println!(
        "{n_enc} encode cases, {n_parse} parse cases ({n_stream} also streamed): {bad} mismatches in {:.2} s",
        t0.elapsed().as_secs_f64()
    );
    if bad > 0 {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("tokcheck") => tokcheck(&args),
        Some("enccheck") => enccheck(&args),
        Some("serve") => serve(&args),
        Some("mock-engine") => mock_engine(&args),
        Some("admin") => admin(&args),
        Some("tokenize") => {
            let path = args.get(2).unwrap_or_else(|| usage());
            let tok = load_tokenizer(&args);
            let text = std::fs::read_to_string(path).expect("read");
            println!("{:?}", tok.encode(&text));
        }
        Some("tokbench") => {
            let path = args.get(2).unwrap_or_else(|| usage());
            let tok = load_tokenizer(&args);
            let text = std::fs::read_to_string(path).expect("read");
            let reference = tok.encode(&text);
            for threads in [1usize, 4, 16] {
                let t0 = Instant::now();
                let ids = tok.encode_parallel(&text, threads);
                let s = t0.elapsed().as_secs_f64();
                println!(
                    "threads {threads}: {} tokens from {:.1} MB in {:.3} s = {:.2} M tok/s{}",
                    ids.len(),
                    text.len() as f64 / 1e6,
                    s,
                    ids.len() as f64 / s / 1e6,
                    if ids == reference { "" } else { "  (DIFFERS from the single-pass encoding)" }
                );
            }
        }
        _ => usage(),
    }
}
