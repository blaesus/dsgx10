// sparkDash 1.8.8 (754f40a) against dsfront, with sparkDash's own code doing the requests and the timing:
//   decode:  its decode bench (DecodeBenchManager: a 32-token warm-up, then one c1 stream of 256 tokens,
//            temperature 0, top_p 1, thinking off, min_tokens/ignore_eos/stop []): two discarded jobs, then
//            five rounds of prose, code, structured and json; then its prompt catalogs (TEXT_PROMPTS,
//            STRUCTURAL_PROMPTS, DECODE_CODE_TASKS), each prompt at c1 the same way (the 32-token warm-up,
//            then the 256-token stream) through its runStreamingRequest.
//   prefill: cold prefill on real text: sparkDash's prefill request (8 new tokens, tok/s = usage.prompt_tokens
//            over the time to the first token) with its header and footer around a slice of CORPUS that no
//            other prompt uses, after a 512-token filler warm-up and a ~2k real-text one (discarded); two
//            passes. Then its own filler bench (PrefillBenchManager: one repeated token) twice.
// One JSON line per measured request in OUT (mode 0600), in the order sent; sdreport.py joins them with
// rank 0's log. part.sh runs it on a dev engine in a ring part.
//   SD_ROOT=<sparkDash clone> node sdrun.mjs decode|prefill PORT KEYFILE OUT.jsonl [CORPUS]
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const [mode, portS, keyFile, outPath, corpusPath] = process.argv.slice(2);
const SD = process.env.SD_ROOT;
const imp = (p) => import(pathToFileURL(path.join(SD, p)).href);
const { decodeBenchManager } = await imp("server/collectors/DecodeBench.js");
const { prefillBenchManager, buildPrefillPrompt, timeoutMsForSize } = await imp("server/collectors/PrefillBench.js");
const { runStreamingRequest, applyThinkingFlags, closeLlmStreamAgent } = await imp("server/collectors/LlmStreaming.js");
const P = await imp("src/shared/llmPrompts.js");

const port = Number(portS);
const apiKey = fs.readFileSync(keyFile, "utf8").trim();
const url = `http://127.0.0.1:${port}/v1/chat/completions`;
const out = fs.openSync(outPath, "a", 0o600);
const sha = (s) => crypto.createHash("sha256").update(s || "").digest("hex").slice(0, 16);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), "sdrun:", ...a);
let seq = 0;
const rec = (o) => fs.writeSync(out, JSON.stringify({ seq: seq++, t: Date.now(), ...o }) + "\n");

// a request as sparkDash builds it: decodeRequestBody / prefillRequestBody, plus the wave's fill fields
function body(prompt, maxTokens, fill) {
  const b = {
    model: undefined,
    messages: [{ role: "user", content: prompt }],
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1,
    stream: true,
    stream_options: { include_usage: true },
  };
  applyThinkingFlags(b, null, false);
  if (fill) Object.assign(b, { min_tokens: maxTokens, ignore_eos: true, stop: [] });
  return b;
}
async function request(prompt, maxTokens, fill, timeoutMs = 360_000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await runStreamingRequest(url, body(prompt, maxTokens, fill), ctrl.signal, {
      debug: true,
      collectContent: true,
      retryOnThinking400: true,
      thinking: false,
      apiKey,
    });
  } finally {
    clearTimeout(t);
  }
}
const pick = (r) => ({
  decodeTps: r.decodeTps,
  decodeTokens: r.decodeTokens,
  completionTokens: r.completionTokens,
  promptTokens: r.prefillTokens,
  ttftMs: r.ttftMs,
  decodeMs: r.decodeMs,
  totalMs: r.totalMs,
  prefillTps: r.prefillTps,
  finish: r.finishReason ?? null,
  sse: r.sseEventCount ?? null,
  error: r.error,
});
async function waitJob(mgr, id) {
  let j;
  while ((j = mgr.getJob(id)) && j.status === "running") await sleep(100);
  return j;
}

async function decodeJob(type) {
  const s = decodeBenchManager.start({
    sparkId: "gb10-rank0",
    lanIp: "127.0.0.1",
    port,
    modelId: null,
    concurrencies: [1],
    maxTokens: 256,
    promptType: type,
    debug: true,
    apiKey,
  });
  const j = await waitJob(decodeBenchManager, s.benchId);
  const w = j.results?.[0];
  const st = w?.streams?.[0] || {};
  return {
    status: j.status,
    error: j.error || w?.error || st.error || null,
    medianDecodeTps: w?.medianDecodeTps ?? null,
    decodeTps: st.decodeTps,
    decodeTokens: st.decodeTokens,
    completionTokens: st.completionTokens,
    promptTokens: st.prefillTokens,
    ttftMs: st.ttftMs,
    decodeMs: st.decodeMs,
    totalMs: st.totalMs,
    finish: st.http?.finishReason ?? null,
    sse: st.http?.sseEventCount ?? null,
    text: st.contentPreview ? sha(JSON.stringify(st.contentPreview)) : null,
    chars: st.contentPreview?.chars ?? null,
  };
}

async function decode() {
  for (let i = 1; i <= 2; i++) {
    const r = await decodeJob("prose");
    rec({ phase: "warm", kind: "prose", run: i, ...r });
    log(`warm-up ${i}: prose ${r.decodeTps} tok/s ${r.error || ""}`);
    if (r.error) throw new Error(`warm-up ${i}: ${r.error}`);
  }
  for (let run = 1; run <= 5; run++) {
    for (const kind of ["prose", "code", "structured", "json"]) {
      const r = await decodeJob(kind);
      rec({ phase: "c1", kind, run, ...r });
      log(`c1 ${kind} run ${run}: ${r.decodeTps} tok/s, ${r.completionTokens} tokens, ttft ${r.ttftMs} ms ${r.error || ""}`);
    }
  }
  const cats = [
    ["prose", P.TEXT_PROMPTS],
    ["structural", P.STRUCTURAL_PROMPTS],
    ["code", P.DECODE_CODE_TASKS],
  ];
  for (const [kind, list] of cats) {
    const v = [];
    for (let idx = 0; idx < list.length; idx++) {
      await request(list[idx], 32, false);
      const r = await request(list[idx], 256, true);
      rec({ phase: "catalog", kind, idx, ...pick(r), text: sha(r.content), chars: (r.content || "").length });
      v.push(r.decodeTps);
      if (r.error) log(`catalog ${kind} ${idx}: ${r.error}`);
    }
    log(`catalog ${kind}: ${v.length} prompts, mean ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(1)} tok/s`);
  }
}

async function prefill() {
  const corpus = fs.readFileSync(corpusPath, "utf8");
  let off = 0;
  const real = (chars) => {
    const s = corpus.slice(off, off + chars);
    off += chars;
    // sparkDash's header and footer (buildPrefillPrompt) around real text instead of its filler
    return `[prefill-bench ${crypto.randomUUID()}]\nIgnore the text below. Reply with the single word OK.\n${s}\nReply OK.`;
  };
  const w1 = await request(buildPrefillPrompt(512, crypto.randomUUID()), 8, false);
  rec({ phase: "warm", kind: "filler", target: 512, ...pick(w1) });
  if (w1.error) throw new Error(`warm-up: ${w1.error}`);
  const wchars = 8000;
  const w2 = await request(real(wchars), 8, false);
  rec({ phase: "warm", kind: "real", target: 2048, chars: wchars, ...pick(w2) });
  const cpt = wchars / Math.max(1, w2.prefillTokens - 30);  // chars per token of the corpus
  log(`warm-ups: ${w1.prefillTps} / ${w2.prefillTps} tok/s; ${cpt.toFixed(2)} chars per token`);
  for (let pass = 1; pass <= 2; pass++) {
    for (const target of [4096, 15360, 32768, 61440, 122880]) {
      const chars = Math.round((target - 30) * cpt);
      const r = await request(real(chars), 8, false, timeoutMsForSize(target));
      rec({ phase: "real", pass, target, chars, ...pick(r) });
      log(`real ${pass} ${target}: ${r.prefillTokens} tokens, ttft ${r.ttftMs} ms, ${r.prefillTps} tok/s ${r.error || ""}`);
    }
  }
  log(`corpus used: ${off} of ${corpus.length} chars`);
  for (let pass = 1; pass <= 2; pass++) {
    const s = prefillBenchManager.start({
      sparkId: "gb10-rank0",
      lanIp: "127.0.0.1",
      port,
      modelId: null,
      contextSizes: [4096, 16384, 32768, 65536, 131072],
      apiKey,
    });
    const j = await waitJob(prefillBenchManager, s.benchId);
    for (const row of j.results || []) {
      rec({
        phase: "filler",
        pass,
        target: row.targetTokens,
        promptTokens: row.promptTokens,
        ttftMs: row.ttftMs,
        prefillTps: row.prefillTps,
        completionTokens: row.completionTokens,
        error: row.error,
      });
      log(`filler ${pass} ${row.targetTokens}: ${row.promptTokens} tokens, ${row.prefillTps} tok/s ${row.error || ""}`);
    }
    if (j.status !== "completed") log(`filler job ${pass}: ${j.status} ${j.error || ""}`);
  }
}

try {
  if (mode === "decode") await decode();
  else if (mode === "prefill") await prefill();
  else throw new Error(`mode ${mode}?`);
  log("done");
} catch (e) {
  log(`failed: ${e?.stack || e}`);
  process.exitCode = 1;
} finally {
  fs.closeSync(out);
  await closeLlmStreamAgent(1000).catch(() => {});
  setTimeout(() => process.exit(process.exitCode || 0), 200).unref();
}
