// Runs edit_pi_check.py's cases through pi's own tool code (prepareArguments, validateToolArguments, execute)
// and prints each result: node edit_pi_check.mjs CASES.json WORKDIR > OUT.json   (PI_DIR: pi's package)
import { mkdirSync, readFileSync, rmSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";

const PI = process.env.PI_DIR;
const { createReadToolDefinition } = await import(`${PI}/dist/core/tools/read.js`);
const { createEditToolDefinition } = await import(`${PI}/dist/core/tools/edit.js`);
const { createWriteToolDefinition } = await import(`${PI}/dist/core/tools/write.js`);
const { validateToolArguments } = await import(`${PI}/node_modules/@earendil-works/pi-ai/dist/utils/validation.js`);

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const work = process.argv[3];
const out = [];
function listFiles(d, base = d, acc = {}) {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) listFiles(p, base, acc);
    else acc[p.slice(base.length + 1)] = readFileSync(p, "utf8");
  }
  return acc;
}
for (const [i, c] of cases.entries()) {
  const dir = join(work, `c${i}`);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (const [p, t] of Object.entries(c.files || {})) {
    mkdirSync(dirname(join(dir, p)), { recursive: true });
    writeFileSync(join(dir, p), t);
  }
  const defs = { read: createReadToolDefinition(dir), edit: createEditToolDefinition(dir), write: createWriteToolDefinition(dir) };
  const tool = defs[c.tool];
  let text, err = false;
  try {
    let args = structuredClone(c.args);
    if (tool.prepareArguments) args = tool.prepareArguments(args);
    const v = validateToolArguments(tool, { name: c.tool, arguments: args });
    const r = await tool.execute("id", v, undefined, undefined, {});
    text = r.content.filter((x) => x.type === "text").map((x) => x.text).join("\n");
  } catch (e) {
    text = e instanceof Error ? e.message : String(e);
    err = true;
  }
  out.push({ text, err, files: listFiles(dir) });
}
console.log(JSON.stringify(out));
