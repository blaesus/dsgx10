//! Port of altar-1's `chat_template.jinja` (AikidoSec/altar-1 at 5d591cd98958e4e7e429517887e80ca3bb7ce2c4,
//! rendered by transformers' sandboxed Jinja with `trim_blocks` and `lstrip_blocks`), text only, and the
//! parser for the model's output.
//!
//! altar-1's template is GLM-5.3's (glm53-4xgx10-switchless/frontend/src/encoding.rs ports that one)
//! but for five lines: `visible_text` ends in a plain `else` where GLM-5.3's has `elif content is not
//! none`, so a null content renders as `None` (Python's `str(None)`) instead of nothing; and the
//! tool-result block's sortability check has no `break`s, so it runs over every result and call
//! after the first reason not to sort (same text, same raise-or-not; only which error a malformed
//! block raises first can differ). Both are mirrored below (docs/protocol.md, "The template").
//!
//! Messages stay as JSON values (OpenAI chat format, as the template receives them) and every
//! branch of the template is mirrored, including its Jinja semantics: undefined attributes render
//! as nothing, `{{ x }}` of a non-string is Python's `str(x)`, truthiness is Python's, `tojson` is
//! `json.dumps(x, ensure_ascii=False)` (pyjson.rs), Jinja variables set in a loop body do not carry
//! into the next iteration, and an error wherever the template raises. `altarfront enccheck` compares
//! it token for token with transformers' `apply_chat_template` over cases from
//! tools/gen_encoding_cases.py.
//!
//! The parse side follows vLLM's GLM-4.7 parser (`vllm/parser/glm47_moe.py`, which vLLM also uses
//! for GLM-5.3): reasoning up to `</think>`, `<tool_call>NAME<arg_key>K</arg_key><arg_value>V
//! </arg_value>…</tool_call>` blocks into OpenAI tool calls, with each value a string unless the
//! tool's JSON schema gives the parameter another type (vLLM's `coerce_to_schema_type`).

use crate::pyjson::{dumps as to_json, float_repr, py_str, truthy};
use serde_json::{Map, Value};

pub const PREFIX: &str = "[gMASK]<sop>";
pub const SYSTEM_SP: &str = "<|system|>";
pub const USER_SP: &str = "<|user|>";
pub const ASSISTANT_SP: &str = "<|assistant|>";
pub const OBSERVATION_SP: &str = "<|observation|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const ARG_KEY_START: &str = "<arg_key>";
pub const ARG_KEY_END: &str = "</arg_key>";
pub const ARG_VALUE_START: &str = "<arg_value>";
pub const ARG_VALUE_END: &str = "</arg_value>";
pub const TOOL_RESPONSE_START: &str = "<tool_response>";
pub const TOOL_RESPONSE_END: &str = "</tool_response>";

/// generation_config.json's eos_token_id: `<|endoftext|>`, `<|user|>`, `<|observation|>`.
pub const STOP_IDS: [u32; 3] = [154820, 154827, 154829];
/// Ids from here up are the head's padding rows (and nothing in tokenizer.json): never sampled.
pub const SAMPLEABLE_VOCAB: u32 = 154856;

/// The reasoning efforts the template knows besides its default, `max`.
pub const EFFORTS: [&str; 2] = ["low", "high"];

const TOOLS_HEADER: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_FOOTER: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

#[derive(Debug, Clone, PartialEq)]
pub struct EncodeError(pub String);

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err(s: impl Into<String>) -> EncodeError {
    EncodeError(s.into())
}

/// The template's variables besides `messages`. `None` is an undefined variable.
#[derive(Default, Clone, Copy)]
pub struct RenderOptions<'a> {
    pub tools: Option<&'a Value>,
    pub reasoning_effort: Option<&'a Value>,
    pub clear_thinking: Option<&'a Value>,
    pub add_generation_prompt: bool,
}

// ------------------------------------------------------------------ Jinja semantics

/// `obj.key` on a JSON value: a mapping's item, else undefined (None).
fn attr<'v>(obj: Option<&'v Value>, key: &str) -> Option<&'v Value> {
    obj.and_then(Value::as_object).and_then(|o| o.get(key))
}

/// `x.0`: the first item of a list, the first character of a string (Ok(None) = undefined, as for
/// an empty one). Mappings, numbers and booleans give undefined too, and any attribute of that
/// undefined raises, which callers report with `Err`.
enum First<'v> {
    Item(&'v Value),
    Char,
    Undefined,
    /// an undefined whose attributes raise
    Raises,
}

fn first(v: Option<&Value>) -> First<'_> {
    match v {
        Some(Value::Array(a)) => a.first().map_or(First::Undefined, First::Item),
        Some(Value::String(s)) if !s.is_empty() => First::Char,
        Some(Value::String(_)) | None => First::Undefined,
        Some(_) => First::Raises,
    }
}

/// `x.0.<key>` on the result of `first`: the attribute, undefined, or an error.
fn first_attr<'v>(v: Option<&'v Value>, key: &str) -> Result<Option<&'v Value>, EncodeError> {
    match first(v) {
        First::Item(x) => Ok(attr(Some(x), key)),
        First::Char => Ok(None),
        First::Undefined | First::Raises => Err(err(format!("UndefinedError: undefined has no attribute '{key}'"))),
    }
}

fn is_string(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::String(_)))
}

fn is_mapping(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Object(_)))
}

/// Jinja's `iterable` test: strings, lists, mappings and undefined (which iterates as empty).
fn is_iterable(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::String(_) | Value::Array(_) | Value::Object(_)))
}

fn str_eq(v: Option<&Value>, s: &str) -> bool {
    v.and_then(Value::as_str) == Some(s)
}

/// `{{ x }}`: Python's `str(x)`, nothing for undefined.
fn render_value(v: Option<&Value>) -> String {
    v.map_or(String::new(), py_str)
}

/// Python `==` between two values or undefineds (Jinja's undefined equals another undefined).
fn py_eq(a: Option<&Value>, b: Option<&Value>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => value_eq(x, y),
        _ => false,
    }
}

fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.to_string().parse::<f64>().ok() == y.to_string().parse::<f64>().ok(),
        (Value::Bool(x), Value::Number(n)) | (Value::Number(n), Value::Bool(x)) => n.to_string().parse::<f64>().ok() == Some(*x as u8 as f64),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| value_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| value_eq(v, w))),
        _ => a == b,
    }
}

/// Python `str.isspace()` for one char (what `str.strip()` removes).
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// `str + x` in Jinja: x must be a string.
fn concat_str<'v>(v: Option<&'v Value>, what: &str) -> Result<&'v str, EncodeError> {
    match v {
        Some(Value::String(s)) => Ok(s),
        None => Err(err(format!("UndefinedError: {what} is undefined"))),
        Some(x) => Err(err(format!("TypeError: can only concatenate str (not {}) for {what}", py_type(x)))),
    }
}

fn py_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

// ------------------------------------------------------------------ the template's macros

/// `tool_to_json(tool)`: the tool's items in order, without `defer_loading` and `strict`, keys as
/// they are (unescaped) and values through tojson.
fn tool_to_json(tool: Option<&Value>) -> Result<String, EncodeError> {
    let o = tool.and_then(Value::as_object).ok_or_else(|| err("tool_to_json: a tool must be a mapping"))?;
    let mut s = String::from("{");
    let mut first = true;
    for (k, v) in o {
        if k != "defer_loading" && k != "strict" {
            if !first {
                s.push_str(", ");
            }
            first = false;
            s.push('"');
            s.push_str(k);
            s.push_str("\": ");
            s.push_str(&to_json(v));
        }
    }
    s.push('}');
    Ok(s)
}

/// `if 'function' in tool: tool = tool['function']`
fn unwrap_function(tool: &Value) -> Result<Option<&Value>, EncodeError> {
    match tool {
        Value::Object(o) => Ok(Some(o.get("function").unwrap_or(tool))),
        Value::Array(a) => Ok(if a.iter().any(|x| str_eq(Some(x), "function")) { None } else { Some(tool) }),
        Value::String(_) => Ok(Some(tool)),
        _ => Err(err("TypeError: argument of type is not iterable ('function' in tool)")),
    }
}

fn tools_list(tools: Option<&Value>) -> Result<&[Value], EncodeError> {
    match tools {
        Some(Value::Array(a)) => Ok(a),
        None | Some(Value::Null) => Ok(&[]),
        _ => Err(err("tools must be a list")),
    }
}

fn tool_references_to_response(refs: &Value, o: &RenderOptions) -> Result<String, EncodeError> {
    if !truthy(o.tools) {
        // the macro is only defined when tools are given
        return Err(err("UndefinedError: 'tool_references_to_response' is undefined"));
    }
    let mut s = String::from("<tool_response><tools>\n");
    let refs = refs.as_array().map(Vec::as_slice).unwrap_or(&[]);
    for tr in refs {
        for tool in tools_list(o.tools)? {
            let t = unwrap_function(tool)?;
            if py_eq(attr(t, "name"), attr(Some(tr), "name")) {
                s.push_str(&tool_to_json(t)?);
                s.push('\n');
            }
        }
    }
    s.push_str("</tools></tool_response>");
    Ok(s)
}

const MEDIA: [&str; 7] = ["image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"];

/// `visible_text(content)`
pub fn visible_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => {
            let mut s = String::new();
            for item in items {
                match item {
                    Value::Object(_) if str_eq(attr(Some(item), "type"), "text") => s.push_str(&render_value(attr(Some(item), "text"))),
                    Value::String(t) => s.push_str(t),
                    Value::Object(_) => {
                        if let Some(t) = attr(Some(item), "type").and_then(Value::as_str).filter(|t| MEDIA.contains(t)) {
                            let media = t.replace("_url", "").replace("input_", "");
                            s.push_str(&format!(
                                "<reminder>You are unable to process this {media} because you don't have multi-modal input ability. Try different methods.</reminder>"
                            ));
                        }
                    }
                    _ => {}
                }
            }
            s
        }
        // altar-1: `{%- else -%}{{- content }}` (GLM-5.3: `elif content is not none`): null is "None",
        // only an undefined content renders as nothing
        None => String::new(),
        Some(v) => py_str(v),
    }
}

fn tool_response(text: &str) -> String {
    format!("{TOOL_RESPONSE_START}{text}{TOOL_RESPONSE_END}")
}

/// `id_of(obj)`: tool_call_id, else id, as text ("" when neither is truthy).
fn id_of(obj: Option<&Value>) -> String {
    let t = attr(obj, "tool_call_id");
    if truthy(t) {
        return render_value(t);
    }
    let i = attr(obj, "id");
    if truthy(i) {
        return render_value(i);
    }
    String::new()
}

/// `is_list_of_outputs(m)`
fn is_list_of_outputs(m: &Value) -> Result<bool, EncodeError> {
    let c = attr(Some(m), "content");
    if !truthy(c) {
        return Ok(false);
    }
    Ok(first_attr(c, "output")?.is_some())
}

/// `x is iterable and x is not string and x is not mapping and x and x.0.type == "tool_reference"`
fn is_tool_reference_list(x: Option<&Value>) -> Result<bool, EncodeError> {
    if !(is_iterable(x) && !is_string(x) && !is_mapping(x) && truthy(x)) {
        return Ok(false);
    }
    Ok(str_eq(first_attr(x, "type")?, "tool_reference"))
}

fn output_response(output: Option<&Value>, o: &RenderOptions) -> Result<String, EncodeError> {
    if is_tool_reference_list(output)? {
        tool_references_to_response(output.unwrap(), o)
    } else {
        Ok(tool_response(&visible_text(output)))
    }
}

/// `render_tool_response(m)`
fn render_tool_response(m: &Value, o: &RenderOptions) -> Result<String, EncodeError> {
    let c = attr(Some(m), "content");
    if let Some(Value::String(s)) = c {
        return Ok(tool_response(s));
    }
    if truthy(c) && !is_mapping(c) && str_eq(first_attr(c, "type")?, "tool_reference") {
        return tool_references_to_response(c.unwrap(), o);
    }
    if is_list_of_outputs(m)? {
        let mut s = String::new();
        for tr in c.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
            s.push_str(&output_response(attr(Some(tr), "output"), o)?);
        }
        return Ok(s);
    }
    Ok(tool_response(&visible_text(c)))
}

/// Entries of a list-of-outputs tool message.
fn entries(m: &Value) -> &[Value] {
    attr(Some(m), "content").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// `has_dup_tool_result_id(lo, hi, target)`
fn has_dup_tool_result_id(block: &[Value], target: &str) -> Result<bool, EncodeError> {
    let mut n = 0;
    for m in block {
        if is_list_of_outputs(m)? {
            n += entries(m).iter().filter(|e| id_of(Some(e)) == target).count();
        } else if id_of(Some(m)) == target {
            n += 1;
        }
        if n > 1 {
            break;
        }
    }
    Ok(n > 1)
}

/// The `<|observation|>` block for a run of tool messages; `calls` are the preceding assistant's
/// tool calls, if that message is an assistant with some.
fn render_tool_block(block: &[Value], calls: Option<&[Value]>, o: &RenderOptions) -> Result<String, EncodeError> {
    let mut s = String::from(OBSERVATION_SP);
    let mut can_sort = calls.is_some_and(|c| !c.is_empty());
    if let Some(calls) = calls.filter(|_| can_sort) {
        let tc_exists = |target: &str| calls.iter().any(|tc| id_of(Some(tc)) == target);
        let ok = |id: String| -> Result<bool, EncodeError> {
            Ok(!id.is_empty() && !has_dup_tool_result_id(block, &id)? && tc_exists(&id))
        };
        // altar-1's check has no `break`: every result and every call is looked at even after
        // can_sort is false (GLM-5.3's stops at the first), so the first error a malformed block
        // raises is the template's
        for m in block {
            if is_list_of_outputs(m)? {
                for e in entries(m) {
                    if !ok(id_of(Some(e)))? {
                        can_sort = false;
                    }
                }
            } else if !ok(id_of(Some(m)))? {
                can_sort = false;
            }
        }
        for (i, tc) in calls.iter().enumerate() {
            let id = id_of(Some(tc));
            if id.is_empty() || calls[i + 1..].iter().any(|t| id_of(Some(t)) == id) {
                can_sort = false;
            }
        }
        if can_sort {
            for tc in calls {
                let tc_id = id_of(Some(tc));
                for m in block {
                    if is_list_of_outputs(m)? {
                        for e in entries(m) {
                            if id_of(Some(e)) == tc_id {
                                s.push_str(&output_response(attr(Some(e), "output"), o)?);
                            }
                        }
                    } else if id_of(Some(m)) == tc_id {
                        s.push_str(&render_tool_response(m, o)?);
                    }
                }
            }
            return Ok(s);
        }
    }
    for m in block {
        s.push_str(&render_tool_response(m, o)?);
    }
    Ok(s)
}

/// One tool call of an assistant message: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`.
pub fn render_tool_call(tc: &Value) -> Result<String, EncodeError> {
    let f = attr(Some(tc), "function");
    let tc = if truthy(f) { f } else { Some(tc) };
    let mut s = String::from(TOOL_CALL_START);
    s.push_str(concat_str(attr(tc, "name"), "the tool call's name")?);
    let args = attr(tc, "arguments").and_then(Value::as_object).ok_or_else(|| err("a tool call's arguments must be a mapping"))?;
    for (k, v) in args {
        s.push_str(ARG_KEY_START);
        s.push_str(k);
        s.push_str(ARG_KEY_END);
        s.push_str(ARG_VALUE_START);
        match v {
            Value::String(x) => s.push_str(x),
            _ => s.push_str(&to_json(v)),
        }
        s.push_str(ARG_VALUE_END);
    }
    s.push_str(TOOL_CALL_END);
    Ok(s)
}

/// Python `s.split(sep)[i]` for i = 0 and -1.
fn split_first<'a>(s: &'a str, sep: &str) -> &'a str {
    s.split(sep).next().unwrap()
}
fn split_last<'a>(s: &'a str, sep: &str) -> &'a str {
    s.rsplit(sep).next().unwrap()
}

/// The reasoning and the visible content of an assistant message, as the template reads them.
pub fn assistant_parts(m: &Value) -> (Option<String>, String) {
    let mut content = visible_text(attr(Some(m), "content"));
    let reasoning = match attr(Some(m), "reasoning_content") {
        Some(Value::String(r)) => Some(r.clone()),
        _ if content.contains(THINK_END) => {
            let r = split_last(split_first(&content, THINK_END), THINK_START).to_string();
            content = split_last(&content, THINK_END).to_string();
            Some(r)
        }
        _ => None,
    };
    (reasoning, content)
}

// ------------------------------------------------------------------ the template

/// The prompt text `apply_chat_template(messages, tools=…, add_generation_prompt=…, **kwargs)` gives.
pub fn render(messages: &[Value], o: &RenderOptions) -> Result<String, EncodeError> {
    if messages.is_empty() {
        // apply_chat_template's own check, before the template
        return Err(err("ValueError: Cannot apply chat template to an empty conversation"));
    }
    let mut s = String::from(PREFIX);
    let effort = match o.reasoning_effort {
        Some(Value::String(e)) if EFFORTS.contains(&e.as_str()) => e.as_str(),
        _ => "max",
    };
    s.push_str(SYSTEM_SP);
    s.push_str("Reasoning Effort: ");
    s.push_str(&capitalize(effort));
    let clear_thinking = truthy(o.clear_thinking);
    if truthy(o.tools) {
        s.push_str(TOOLS_HEADER);
        let list = match o.tools {
            Some(Value::Array(a)) => a.as_slice(),
            _ => return Err(err("tools must be a list")),
        };
        for tool in list {
            let t = unwrap_function(tool)?;
            if !truthy(attr(t, "defer_loading")) {
                s.push_str(&tool_to_json(t)?);
                s.push('\n');
            }
        }
        s.push_str(TOOLS_FOOTER);
    }
    for (i, m) in messages.iter().enumerate() {
        if !m.is_object() {
            return Err(err(format!("messages[{i}] is not a mapping")));
        }
    }
    let last_user = messages.iter().rposition(|m| str_eq(attr(Some(m), "role"), "user"));
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        let role = attr(Some(m), "role").and_then(Value::as_str).unwrap_or("");
        match role {
            "user" => {
                s.push_str(USER_SP);
                s.push_str(&visible_text(attr(Some(m), "content")));
            }
            "assistant" => {
                s.push_str(ASSISTANT_SP);
                let (reasoning, content) = assistant_parts(m);
                let keep = !clear_thinking || last_user.map_or(true, |u| i > u);
                match reasoning {
                    Some(r) if keep => {
                        s.push_str(THINK_START);
                        s.push_str(&r);
                        s.push_str(THINK_END);
                    }
                    _ => s.push_str("<think></think>"),
                }
                let c = py_strip(&content);
                if !c.is_empty() {
                    s.push_str(c);
                }
                let tcs = attr(Some(m), "tool_calls");
                if truthy(tcs) {
                    let list = tcs.and_then(Value::as_array).ok_or_else(|| err("tool_calls must be a list"))?;
                    for tc in list {
                        s.push_str(&render_tool_call(tc)?);
                    }
                }
            }
            "tool" => {
                if i == 0 || !str_eq(attr(Some(&messages[i - 1]), "role"), "tool") {
                    let mut end = i;
                    while end + 1 < messages.len() && str_eq(attr(Some(&messages[end + 1]), "role"), "tool") {
                        end += 1;
                    }
                    let calls = if i > 0 && str_eq(attr(Some(&messages[i - 1]), "role"), "assistant") {
                        let tcs = attr(Some(&messages[i - 1]), "tool_calls");
                        if truthy(tcs) {
                            Some(tcs.and_then(Value::as_array).ok_or_else(|| err("tool_calls must be a list"))?.as_slice())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    s.push_str(&render_tool_block(&messages[i..=end], calls, o)?);
                }
            }
            "system" => {
                s.push_str(SYSTEM_SP);
                s.push_str(&visible_text(attr(Some(m), "content")));
            }
            _ => {}
        }
        i += 1;
    }
    if o.add_generation_prompt {
        s.push_str(ASSISTANT_SP);
        s.push_str(THINK_START);
    }
    Ok(s)
}

/// Jinja's `capitalize`: the first character upper case, the rest lower case.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c.flat_map(char::to_lowercase)).collect(),
        None => String::new(),
    }
}

// ------------------------------------------------------------------ tool-call arguments (vLLM's rules)

/// The JSON-schema types a parameter allows (vLLM `extract_types_from_schema`).
fn schema_types(schema: &Value) -> Vec<String> {
    let Some(o) = schema.as_object() else { return vec!["string".into()] };
    let mut types: Vec<String> = Vec::new();
    let mut add = |t: &str| {
        if !types.iter().any(|x| x == t) {
            types.push(t.to_string());
        }
    };
    match o.get("type") {
        Some(Value::String(t)) => add(t),
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).for_each(&mut add),
        _ => {}
    }
    if let Some(Value::Array(e)) = o.get("enum") {
        for v in e {
            add(match v {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if !n.to_string().contains(['.', 'e', 'E']) => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            });
        }
    }
    for k in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(cs)) = o.get(k) {
            for c in cs {
                for t in schema_types(c) {
                    add(&t);
                }
            }
        }
    }
    if types.is_empty() {
        types.push("string".into());
    }
    types
}

fn type_alias(t: &str) -> String {
    let t = t.trim_matches(py_isspace).to_lowercase();
    match t.as_str() {
        "str" | "text" | "varchar" | "char" | "enum" => "string",
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short" | "unsigned" => "integer",
        "float" | "float32" | "float64" | "double" => "number",
        "bool" => "boolean",
        "dict" => "object",
        "arr" | "list" | "sequence" => "array",
        _ => return t,
    }
    .to_string()
}

/// Digits with PEP 515 underscores (single, between digits) removed; None if malformed.
fn py_digits(s: &str) -> Option<String> {
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return None;
    }
    let d: String = s.chars().filter(|&c| c != '_').collect();
    d.chars().all(|c| c.is_ascii_digit()).then_some(d)
}

/// Python `int(s)` for a str (ASCII digits), as JSON number text.
fn py_int(s: &str) -> Option<String> {
    let t = s.trim_matches(py_isspace);
    let (neg, body) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let d = py_digits(body)?;
    let d = d.trim_start_matches('0');
    Some(if d.is_empty() { "0".into() } else if neg { format!("-{d}") } else { d.to_string() })
}

/// Python `float(s)` for a str: finite values only (None where Python raises or gives inf/nan).
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim_matches(py_isspace);
    let (sign, body) = match t.as_bytes().first()? {
        b'-' => ("-", &t[1..]),
        b'+' => ("", &t[1..]),
        _ => ("", t),
    };
    // mantissa [. fraction] [e exponent], each digit run with PEP 515 underscores
    let (mant, exp) = match body.find(['e', 'E']) {
        Some(p) => (&body[..p], Some(&body[p + 1..])),
        None => (body, None),
    };
    let (int_part, frac) = match mant.find('.') {
        Some(p) => (&mant[..p], Some(&mant[p + 1..])),
        None => (mant, None),
    };
    let ip = if int_part.is_empty() { Some(String::new()) } else { py_digits(int_part) }?;
    let fp = match frac {
        None => String::new(),
        Some("") => String::new(),
        Some(f) => py_digits(f)?,
    };
    if ip.is_empty() && fp.is_empty() {
        return None;
    }
    let ep = match exp {
        None => String::new(),
        Some(e) => {
            let (es, eb) = match e.as_bytes().first()? {
                b'-' => ("-", &e[1..]),
                b'+' => ("", &e[1..]),
                _ => ("", e),
            };
            format!("e{es}{}", py_digits(eb)?)
        }
    };
    let f: f64 = format!("{sign}{}.{}{ep}", if ip.is_empty() { "0" } else { &ip }, if fp.is_empty() { "0" } else { &fp }).parse().ok()?;
    f.is_finite().then_some(f)
}

/// Every number in a parsed JSON value is finite (Python's json takes NaN and Infinity).
fn json_finite(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.to_string().parse::<f64>().is_ok_and(f64::is_finite),
        Value::Array(a) => a.iter().all(json_finite),
        Value::Object(o) => o.values().all(json_finite),
        _ => true,
    }
}

fn num(text: String) -> Value {
    Value::Number(serde_json::from_str(&text).expect("number text"))
}

/// vLLM `coerce_to_schema_type`: the first type in priority order the raw string converts to.
pub fn coerce(value: &str, types: &[String]) -> Value {
    let types: Vec<String> = types.iter().map(|t| type_alias(t)).collect();
    let has = |t: &str| types.iter().any(|x| x == t);
    for t in ["null", "integer", "number", "boolean", "object", "array", "string"] {
        if !has(t) {
            continue;
        }
        match t {
            "null" if value.to_lowercase() == "null" => return Value::Null,
            "string" => return Value::String(value.into()),
            "integer" => {
                if let Some(i) = py_int(value) {
                    return num(i);
                }
            }
            "number" => {
                if let Some(f) = py_float(value) {
                    return if f.fract() != 0.0 {
                        num(float_repr(f))
                    } else {
                        let s = format!("{f:.0}");
                        num(if s == "-0" { "0".into() } else { s })
                    };
                }
            }
            "boolean" => match value.to_lowercase().trim_matches(py_isspace) {
                "true" | "1" => return Value::Bool(true),
                "false" | "0" => return Value::Bool(false),
                _ => {}
            },
            "object" | "array" => {
                if let Some(v) = crate::pyjson::loads(value).filter(json_finite) {
                    return v;
                }
            }
            _ => {}
        }
    }
    match crate::pyjson::loads(value).filter(json_finite) {
        Some(v) => v,
        None => Value::String(value.into()),
    }
}

/// `<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>` (DOTALL), every match in order:
/// (key stripped, raw value). A key seen twice keeps its first place and its last value.
pub fn parse_args(raw: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut from = 0;
    while let Some(p) = raw[from..].find(ARG_KEY_START) {
        let a = from + p;
        let ks = a + ARG_KEY_START.len();
        let mut found = None;
        let mut kfrom = ks;
        while let Some(q) = raw[kfrom..].find(ARG_KEY_END) {
            let b = kfrom + q;
            let after = &raw[b + ARG_KEY_END.len()..];
            let ws = after.len() - after.trim_start_matches(py_isspace).len();
            let c = b + ARG_KEY_END.len() + ws;
            if raw[c..].starts_with(ARG_VALUE_START) {
                let vs = c + ARG_VALUE_START.len();
                match raw[vs..].find(ARG_VALUE_END) {
                    Some(r) => found = Some((b, vs, vs + r)),
                    None => {}
                }
                break;
            }
            kfrom = b + 1;
        }
        match found {
            Some((b, vs, ve)) => {
                let key = py_strip(&raw[ks..b]).to_string();
                let val = raw[vs..ve].to_string();
                match out.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = val,
                    None => out.push((key, val)),
                }
                from = ve + ARG_VALUE_END.len();
            }
            None => from = a + 1,
        }
    }
    out
}

/// The tools a request offered, for parsing calls: name → the parameters' `properties`.
#[derive(Clone, Debug, Default)]
pub struct ToolSchemas {
    tools: Vec<(String, Option<Map<String, Value>>)>,
}

impl ToolSchemas {
    /// From the request's tools (OpenAI form, or bare functions).
    pub fn from_tools(tools: &[Value]) -> Self {
        let mut out = Vec::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            if let Some(name) = f.get("name").and_then(Value::as_str) {
                let props = f.get("parameters").and_then(|p| p.get("properties")).and_then(Value::as_object).cloned();
                out.push((name.to_string(), props));
            }
        }
        ToolSchemas { tools: out }
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    fn find(&self, name: &str) -> Option<&Option<Map<String, Value>>> {
        self.tools.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }
}

/// A complete `<tool_call>` block's inside (between the tags) as (name, arguments JSON), or None
/// where vLLM would not accept it (no name, or a name the request did not offer).
pub fn parse_tool_call(inner: &str, tools: &ToolSchemas) -> Option<(String, String)> {
    let (name, raw) = match inner.find(ARG_KEY_START) {
        Some(p) => (&inner[..p], &inner[p..]),
        None => (inner, ""),
    };
    let name = py_strip(name);
    if name.is_empty() {
        return None;
    }
    let props = tools.find(name)?;
    let mut args = Map::new();
    for (k, v) in parse_args(raw) {
        let schema = props.as_ref().and_then(|p| p.get(&k)).filter(|s| s.is_object());
        let val = match schema {
            Some(s) => coerce(&v, &schema_types(s)),
            None => Value::String(v),
        };
        args.insert(k, val);
    }
    Some((name.to_string(), to_json(&Value::Object(args))))
}

// ------------------------------------------------------------------ streaming parser

/// What the model has said so far, split the way the API reports it.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A complete tool call: (index, name, arguments JSON).
    ToolCall(usize, String, String),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Reasoning,
    Content,
    /// Inside `<tool_call>`, waiting for `</tool_call>`.
    Call,
    Done,
}

/// Incremental parser for streamed output (the prompt ends with `<|assistant|><think>`, so the
/// output starts in the reasoning). Feed decoded text as it arrives: reasoning and content come out
/// as soon as they cannot be the start of a marker, each tool call once its block is closed. As in
/// vLLM: `</think>` ends the reasoning, a `<tool_call>` inside the reasoning ends it too, a stray
/// `<think>` in the reasoning and a stray `</think>` in the content are dropped, and `<think>` in the
/// content opens reasoning again. A block that does not parse as a call of an offered tool, or is
/// cut off, is passed on as content.
pub struct StreamParser {
    phase: Phase,
    buf: String,
    calls: usize,
    tools: Option<ToolSchemas>,
}

impl StreamParser {
    /// `thinking`: the prompt ends in `<think>` (always, with the template's generation prompt).
    /// `tools`: what the request offered; None or empty means tool-call text is plain content.
    pub fn new(thinking: bool, tools: Option<ToolSchemas>) -> Self {
        StreamParser {
            phase: if thinking { Phase::Reasoning } else { Phase::Content },
            buf: String::new(),
            calls: 0,
            tools: tools.filter(|t| !t.is_empty()),
        }
    }

    pub fn tool_calls_seen(&self) -> usize {
        self.calls
    }

    pub fn push(&mut self, text: &str, out: &mut Vec<Delta>) {
        if self.phase == Phase::Done {
            return;
        }
        self.buf.push_str(text);
        self.advance(out, false);
    }

    /// End of generation: flush everything that is left.
    pub fn finish(&mut self, out: &mut Vec<Delta>) {
        self.advance(out, true);
        let rest = std::mem::take(&mut self.buf);
        match self.phase {
            Phase::Reasoning if !rest.is_empty() => push_delta(out, Delta::Reasoning(rest)),
            Phase::Content if !rest.is_empty() => push_delta(out, Delta::Content(rest)),
            // cut off inside a call (max_tokens or a stop): the raw text
            Phase::Call => push_delta(out, Delta::Content(format!("{TOOL_CALL_START}{rest}"))),
            _ => {}
        }
        self.phase = Phase::Done;
    }

    /// Length of the longest suffix of `s` that is a proper prefix of one of `markers`.
    fn holdback(s: &str, markers: &[&str]) -> usize {
        let mut best = 0;
        for m in markers {
            let max = m.len().saturating_sub(1).min(s.len());
            for k in (1..=max).rev() {
                if s.is_char_boundary(s.len() - k) && m.starts_with(&s[s.len() - k..]) {
                    best = best.max(k);
                    break;
                }
            }
        }
        best
    }

    /// The earliest of `markers` in the buffer: (position, marker).
    fn earliest(&self, markers: &[&'static str]) -> Option<(usize, &'static str)> {
        let mut best: Option<(usize, &'static str)> = None;
        for m in markers {
            if let Some(p) = self.buf.find(m) {
                if best.map_or(true, |(bp, _)| p < bp) {
                    best = Some((p, m));
                }
            }
        }
        best
    }

    fn take(&mut self, n: usize) -> String {
        let s = self.buf[..n].to_string();
        self.buf.drain(..n);
        s
    }

    fn advance(&mut self, out: &mut Vec<Delta>, at_end: bool) {
        loop {
            match self.phase {
                Phase::Reasoning | Phase::Content => {
                    let reasoning = self.phase == Phase::Reasoning;
                    let markers: &[&'static str] = match (reasoning, self.tools.is_some()) {
                        (true, true) => &[THINK_END, THINK_START, TOOL_CALL_START],
                        (true, false) => &[THINK_END, THINK_START],
                        (false, true) => &[TOOL_CALL_START, THINK_START, THINK_END],
                        (false, false) => &[THINK_START, THINK_END],
                    };
                    let wrap = |s: String| if reasoning { Delta::Reasoning(s) } else { Delta::Content(s) };
                    match self.earliest(markers) {
                        Some((p, m)) => {
                            if p > 0 {
                                let s = self.take(p);
                                push_delta(out, wrap(s));
                            }
                            self.buf.drain(..m.len());
                            self.phase = match m {
                                THINK_END => Phase::Content,
                                THINK_START => Phase::Reasoning,
                                _ => Phase::Call,
                            };
                        }
                        None => {
                            let keep = if at_end { 0 } else { Self::holdback(&self.buf, markers) };
                            let n = self.buf.len() - keep;
                            if n > 0 {
                                let s = self.take(n);
                                push_delta(out, wrap(s));
                            }
                            return;
                        }
                    }
                }
                Phase::Call => {
                    let Some(p) = self.buf.find(TOOL_CALL_END) else { return };
                    let inner = self.take(p);
                    self.buf.drain(..TOOL_CALL_END.len());
                    match parse_tool_call(&inner, self.tools.as_ref().unwrap()) {
                        Some((name, args)) => {
                            out.push(Delta::ToolCall(self.calls, name, args));
                            self.calls += 1;
                        }
                        None => push_delta(out, Delta::Content(format!("{TOOL_CALL_START}{inner}{TOOL_CALL_END}"))),
                    }
                    self.phase = Phase::Content;
                }
                Phase::Done => {
                    self.buf.clear();
                    return;
                }
            }
        }
    }
}

/// Append a text delta, merging it into the previous one of the same kind.
fn push_delta(out: &mut Vec<Delta>, d: Delta) {
    match (out.last_mut(), &d) {
        (Some(Delta::Reasoning(a)), Delta::Reasoning(b)) | (Some(Delta::Content(a)), Delta::Content(b)) => a.push_str(b),
        _ => out.push(d),
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedMessage {
    pub reasoning_content: String,
    pub content: String,
    /// (name, arguments as a JSON string)
    pub tool_calls: Vec<(String, String)>,
}

/// The whole output at once (the streaming parser fed in one piece).
pub fn parse_completion(text: &str, thinking: bool, tools: Option<ToolSchemas>) -> ParsedMessage {
    let mut p = StreamParser::new(thinking, tools);
    let mut out = Vec::new();
    p.push(text, &mut out);
    p.finish(&mut out);
    let mut m = ParsedMessage::default();
    for d in out {
        match d {
            Delta::Reasoning(s) => m.reasoning_content.push_str(&s),
            Delta::Content(s) => m.content.push_str(&s),
            Delta::ToolCall(_, n, a) => m.tool_calls.push((n, a)),
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minimal_and_example() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let o = RenderOptions { add_generation_prompt: true, ..Default::default() };
        assert_eq!(render(&msgs, &o).unwrap(), "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>");
        let low = json!("low");
        let o = RenderOptions { reasoning_effort: Some(&low), ..o };
        assert!(render(&msgs, &o).unwrap().starts_with("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>"));
        let msgs: Vec<Value> = serde_json::from_value(json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "  c  ", "reasoning_content": "R", "tool_calls": [
                {"id": "1", "function": {"name": "f", "arguments": {"a": 1, "b": "s"}}},
                {"id": "2", "function": {"name": "g", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "2", "content": "r2"},
            {"role": "tool", "tool_call_id": "1", "content": "r1"}]))
        .unwrap();
        let p = render(&msgs, &RenderOptions::default()).unwrap();
        assert!(p.ends_with("<|user|>a<|assistant|><think>R</think>c<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>s</arg_value></tool_call><tool_call>g</tool_call><|observation|><tool_response>r1</tool_response><tool_response>r2</tool_response>"), "{p}");
    }

    #[test]
    fn parse_calls_streamed() {
        let tools = ToolSchemas::from_tools(&[json!({"type": "function", "function": {"name": "read", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "n": {"type": "integer"}, "x": {"type": ["number", "null"]}}}}})]);
        let text = "Let me look.</think>Checking.<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>a.txt</arg_value><arg_key>n</arg_key><arg_value>3</arg_value><arg_key>x</arg_key><arg_value>2.0</arg_value><arg_key>y</arg_key><arg_value>7</arg_value></tool_call><tool_call>nope</tool_call>";
        let whole = parse_completion(text, true, Some(tools.clone()));
        assert_eq!(whole.reasoning_content, "Let me look.");
        assert_eq!(whole.content, "Checking.<tool_call>nope</tool_call>");
        assert_eq!(whole.tool_calls, vec![("read".to_string(), "{\"path\": \"a.txt\", \"n\": 3, \"x\": 2, \"y\": \"7\"}".to_string())]);
        let mut p = StreamParser::new(true, Some(tools));
        let mut out = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for ch in chars.chunks(3) {
            p.push(&ch.iter().collect::<String>(), &mut out);
        }
        p.finish(&mut out);
        let (mut r, mut c, mut calls) = (String::new(), String::new(), Vec::new());
        for d in out {
            match d {
                Delta::Reasoning(s) => r.push_str(&s),
                Delta::Content(s) => c.push_str(&s),
                Delta::ToolCall(_, n, a) => calls.push((n, a)),
            }
        }
        assert_eq!((r, c, calls), (whole.reasoning_content, whole.content, whole.tool_calls));
    }

    #[test]
    fn coercion_like_vllm() {
        let t = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(coerce("5", &t(&["integer"])), json!(5));
        assert_eq!(coerce(" +0_7 ", &t(&["int"])), json!(7));
        assert_eq!(coerce("2.5", &t(&["number"])), json!(2.5));
        assert_eq!(to_json(&coerce("1e20", &t(&["number"]))), "100000000000000000000");
        assert_eq!(coerce("inf", &t(&["number"])), json!("inf"));
        assert_eq!(coerce("True", &t(&["boolean"])), json!(true));
        assert_eq!(coerce("NULL", &t(&["null", "string"])), Value::Null);
        assert_eq!(coerce("[1, 2]", &t(&["array"])), json!([1, 2]));
        assert_eq!(coerce("abc", &t(&["integer"])), json!("abc"));
        assert_eq!(coerce("12", &t(&["whatever"])), json!(12));
        assert_eq!(coerce("12", &t(&["string", "integer"])), json!(12));
    }
}
