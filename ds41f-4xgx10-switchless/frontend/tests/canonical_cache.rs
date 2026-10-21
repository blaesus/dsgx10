//! No checkpoint or serving endpoint is needed: a byte BPE with one merge is sufficient
//! to expose generated IDs that decode correctly but are not canonical input IDs.
use dsfront::{session::SessionTable, tokenizer::{bytes_to_unicode, Tokenizer}};
use serde_json::{json, Map, Value};

fn tokenizer() -> Tokenizer {
    let mut vocab = Map::new();
    for (id, ch) in bytes_to_unicode().iter().enumerate() {
        vocab.insert(ch.to_string(), json!(id));
    }
    vocab.insert("ab".into(), json!(256));
    let added: Vec<Value> = ["<assistant>", "<user>"].iter().enumerate().map(|(i, text)| {
        json!({"id": 257+i, "content": text, "special": true, "normalized": false})
    }).collect();
    Tokenizer::from_json(&json!({"added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab, "merges": ["a b"]}}).to_string()).unwrap()
}

fn prompt() -> String { format!("{}<assistant>", "A synthetic shared prefix. ".repeat(8)) }

#[test]
fn generated_split_ids_never_become_next_input() {
    let tok = tokenizer();
    let text = prompt();
    let mut warm = SessionTable::new(8, 1 << 20);
    let (first, _) = warm.plan(&text, &tok, 1);
    let generated = [b'a' as u32, b'b' as u32];
    assert_eq!(tok.decode(&generated), "ab");
    assert_eq!(tok.encode("ab"), vec![256]);
    warm.finish(first.session_id, &text, &first.ids, &generated, &tok);
    let next = format!("{text}ab<user>next<assistant>");
    let (plan, _) = warm.plan(&next, &tok, 2);
    let (cold, _) = SessionTable::new(8, 1 << 20).plan(&next, &tok, 1);
    assert_eq!(plan.ids, cold.ids, "warm and cold input must match token for token");
    assert_eq!(plan.ids, tok.encode(&next));
    assert_eq!(plan.session_id, first.session_id);
    assert_eq!(plan.reused, first.ids.len(), "reuse must stop before split generated IDs");
    assert_eq!(plan.common, plan.reused);
}

#[test]
fn busy_session_forks_without_copying_noncanonical_ids() {
    let tok = tokenizer();
    let text = prompt();
    let mut table = SessionTable::new(8, 1 << 20);
    let (first, _) = table.plan(&text, &tok, 1);
    table.finish(first.session_id, &text, &first.ids, &[97, 98], &tok);
    let next = format!("{text}ab<user>next<assistant>");
    let (busy, _) = table.plan(&next, &tok, 1);
    let (fork, _) = table.plan(&next, &tok, 3);
    assert_eq!(fork.base_session_id, busy.session_id);
    assert_ne!(fork.session_id, busy.session_id);
    assert_eq!(fork.ids, tok.encode(&next));
    assert_eq!(fork.reused, first.ids.len());
}

#[test]
fn canonical_generated_prefix_remains_reusable() {
    let tok = tokenizer();
    let text = prompt();
    let mut table = SessionTable::new(8, 1 << 20);
    let (first, _) = table.plan(&text, &tok, 1);
    table.finish(first.session_id, &text, &first.ids, &[256], &tok);
    let next = format!("{text}ab<user>next<assistant>");
    let (plan, _) = table.plan(&next, &tok, 1);
    assert_eq!(plan.ids, tok.encode(&next));
    assert_eq!(plan.reused, first.ids.len() + 1);
}
