//! The gate in front of the engine. A request passes while the engine is connected and the ring is
//! not lent; otherwise it waits, up to the hold limit, and is then answered 503 with Retry-After.
//!
//! service.sh lends the ring to a development job through the admin socket (docs/engine.md, "Launch
//! and stop"): `hold IDLE_S` succeeds only when no request is in flight or waiting and none has come
//! for IDLE_S seconds, and from then on requests wait here; the engine stops, the job runs, and
//! `release` once the ring is back lets them through as soon as the engine is connected again. A
//! request in flight is never cut: the hold waits for it.

use crate::engine::EngineClient;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

/// What a request answered 503 is told to wait (Retry-After, seconds).
pub const RETRY_AFTER_S: u64 = 60;
const POLL: Duration = Duration::from_millis(200);

struct State {
    /// requests wait: the ring is lent (since then)
    held: Option<Instant>,
    /// the server is shutting down: waiting requests are answered at once
    closing: bool,
    in_flight: u32,
    waiting: u32,
    /// the last request's arrival, start or end, whether it passed or not
    last: Instant,
    /// since the start: requests that waited and passed, and those answered 503
    passed: u64,
    refused: u64,
}

pub struct Gate {
    st: Mutex<State>,
    hold: Duration,
}

/// A request in flight: dropping it counts the request done.
pub struct Pass {
    gate: Arc<Gate>,
}

impl Drop for Pass {
    fn drop(&mut self) {
        let mut s = self.gate.st.lock().unwrap();
        s.in_flight -= 1;
        s.last = Instant::now();
    }
}

/// A request waiting at the gate; counted out when it passes, is refused, or its client goes away.
struct Waiting {
    gate: Arc<Gate>,
    counted: bool,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if self.counted {
            // a request that waited and left (refused, or its client gone) was activity too
            let mut s = self.gate.st.lock().unwrap();
            s.waiting -= 1;
            s.last = Instant::now();
        }
    }
}

/// Why a request did not pass.
#[derive(Debug, PartialEq)]
pub enum Refused {
    /// it waited the hold limit
    Held(Duration),
    /// the server is shutting down
    Closing,
}

impl Gate {
    pub fn new(hold: Duration) -> Arc<Gate> {
        Arc::new(Gate {
            st: Mutex::new(State { held: None, closing: false, in_flight: 0, waiting: 0, last: Instant::now(), passed: 0, refused: 0 }),
            hold,
        })
    }

    pub fn hold_limit(&self) -> Duration {
        self.hold
    }

    /// Wait until the request may go to the engine: the ring is not lent and the engine is connected.
    pub async fn enter(self: &Arc<Gate>, engine: &EngineClient) -> Result<Pass, Refused> {
        self.enter_when(|| engine.info().is_some()).await
    }

    async fn enter_when(self: &Arc<Gate>, connected: impl Fn() -> bool) -> Result<Pass, Refused> {
        let t0 = Instant::now();
        let mut w = Waiting { gate: self.clone(), counted: false };
        loop {
            {
                let mut s = self.st.lock().unwrap();
                if s.closing {
                    return Err(Refused::Closing);
                }
                if s.held.is_none() && connected() {
                    s.in_flight += 1;
                    s.last = Instant::now();
                    if w.counted {
                        w.counted = false;
                        s.waiting -= 1;
                        s.passed += 1;
                    }
                    return Ok(Pass { gate: self.clone() });
                }
                if t0.elapsed() >= self.hold {
                    s.refused += 1;
                    return Err(Refused::Held(t0.elapsed()));
                }
                if !w.counted {
                    // a request that waits is activity: without this, one that left before the
                    // engine came back let the ring be lent again at once
                    w.counted = true;
                    s.waiting += 1;
                    s.last = Instant::now();
                }
            }
            tokio::time::sleep(POLL).await;
        }
    }

    /// Requests wait from now on, if none is in flight or waiting and none came for `idle`.
    pub fn hold(&self, idle: Duration) -> Result<(), String> {
        let mut s = self.st.lock().unwrap();
        if s.held.is_some() {
            return Ok(());
        }
        if s.in_flight > 0 || s.waiting > 0 || s.last.elapsed() < idle {
            return Err(format!("busy: in_flight {} waiting {} idle_s {}", s.in_flight, s.waiting, s.last.elapsed().as_secs()));
        }
        s.held = Some(Instant::now());
        eprintln!("gate: holding requests (the ring is lent; no request for {} s)", s.last.elapsed().as_secs());
        Ok(())
    }

    pub fn release(&self) {
        let mut s = self.st.lock().unwrap();
        if let Some(t) = s.held.take() {
            eprintln!("gate: released after {} s; {} requests waiting", t.elapsed().as_secs(), s.waiting);
        }
    }

    /// Shutting down: waiting and new requests are answered at once.
    pub fn close(&self) {
        self.st.lock().unwrap().closing = true;
    }

    pub fn status(&self, connected: bool) -> String {
        let s = self.st.lock().unwrap();
        format!(
            "held {} in_flight {} waiting {} idle_s {} engine {} held_s {} passed {} refused {}",
            s.held.is_some() as u8,
            s.in_flight,
            s.waiting,
            s.last.elapsed().as_secs(),
            connected as u8,
            s.held.map_or(0, |t| t.elapsed().as_secs()),
            s.passed,
            s.refused
        )
    }

    /// Whether a request would pass now, without waiting: the ring not lent, the engine connected.
    pub fn open(&self, connected: bool) -> bool {
        let s = self.st.lock().unwrap();
        !s.closing && s.held.is_none() && connected
    }

    /// Whether requests wait because the ring is lent (for /health).
    pub fn held(&self) -> bool {
        self.st.lock().unwrap().held.is_some()
    }

    /// One admin command: `status`, `hold IDLE_S` or `release`; its one-line answer.
    pub fn command(&self, line: &str, connected: bool) -> String {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w[..] {
            ["status"] => self.status(connected),
            ["hold", n] => match n.parse::<u64>() {
                Ok(n) => self.hold(Duration::from_secs(n)).map_or_else(|e| e, |()| "ok".into()),
                Err(_) => "error: hold IDLE_S (seconds)".into(),
            },
            ["release"] => {
                self.release();
                "ok".into()
            }
            _ => "error: status | hold IDLE_S | release".into(),
        }
    }
}

/// Serve the admin socket (mode 600; one command per connection, one line back).
pub async fn admin(gate: Arc<Gate>, engine: EngineClient, path: PathBuf) -> std::io::Result<()> {
    // a socket left by an earlier run goes; anything else there is an error
    match std::fs::symlink_metadata(&path) {
        Ok(m) if m.file_type().is_socket() => std::fs::remove_file(&path)?,
        Ok(_) => return Err(std::io::Error::other(format!("{} exists and is not a socket", path.display()))),
        Err(_) => {}
    }
    let l = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    eprintln!("admin socket {}", path.display());
    loop {
        let (s, _) = l.accept().await?;
        let (gate, engine) = (gate.clone(), engine.clone());
        tokio::spawn(async move {
            let (rd, mut wr) = s.into_split();
            let mut line = String::new();
            let mut rd = BufReader::new(rd.take(256));
            let got = tokio::time::timeout(Duration::from_secs(5), rd.read_line(&mut line)).await;
            let reply = match got {
                Ok(Ok(_)) => gate.command(line.trim(), engine.info().is_some()),
                _ => "error: no command".into(),
            };
            let _ = wr.write_all(format!("{reply}\n").as_bytes()).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn passes_holds_and_refuses() {
        let g = Gate::new(Duration::from_millis(600));
        let up = Arc::new(AtomicBool::new(true));
        let on = {
            let up = up.clone();
            move || up.load(Ordering::SeqCst)
        };
        // a request in flight: no hold
        let p = g.enter_when(&on).await.unwrap();
        assert!(g.hold(Duration::ZERO).unwrap_err().starts_with("busy: in_flight 1"));
        drop(p);
        // it just ended: not idle for a minute
        assert!(g.hold(Duration::from_secs(60)).is_err());
        assert_eq!(g.command("hold 0", true), "ok");
        assert!(g.status(true).starts_with("held 1 in_flight 0 waiting 0"));
        assert!(g.held());
        // held: a request waits, and passes once released
        let (g2, on2) = (g.clone(), on.clone());
        let t = tokio::spawn(async move { g2.enter_when(&on2).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(g.status(true).contains("waiting 1"));
        assert!(g.hold(Duration::ZERO).is_ok(), "held stays held");
        assert_eq!(g.command("release", true), "ok");
        assert_eq!(t.await.unwrap(), Ok(()));
        assert!(g.status(true).contains("waiting 0") && g.status(true).contains("passed 1"));
        // the engine away: a request waits the hold limit, then is refused
        up.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        assert!(matches!(g.enter_when(&on).await, Err(Refused::Held(_))));
        assert!(t0.elapsed() >= Duration::from_millis(600));
        assert!(g.status(false).contains("refused 1") && g.status(false).contains("waiting 0"));
        // a waiting request whose client goes away is counted out
        let (g3, on3) = (g.clone(), on.clone());
        let t = tokio::spawn(async move { g3.enter_when(&on3).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(g.status(false).contains("waiting 1"));
        t.abort();
        let _ = t.await;
        assert!(g.status(false).contains("waiting 0"));
        // and it was activity: no hold for `idle` after it
        assert!(g.hold(Duration::from_millis(200)).is_err(), "a request that waited and left is recent activity");
        // shutting down: answered at once
        g.close();
        assert_eq!(g.enter_when(&on).await.err(), Some(Refused::Closing));
        assert!(g.command("hold x", true).starts_with("error"));
        assert!(g.command("bogus", true).starts_with("error"));
    }
}
