//! dsgb10 front-end: tokeniser, chat encoding, OpenAI-compatible server and the rank-0 protocol.

pub mod encoding;
pub mod engine;
pub mod gate;
pub mod guard;
pub mod mock;
pub mod protocol;
pub mod pyjson;
pub mod server;
pub mod session;
pub mod tokenizer;

pub mod vision;
pub mod feature_file;
pub mod sha256;
