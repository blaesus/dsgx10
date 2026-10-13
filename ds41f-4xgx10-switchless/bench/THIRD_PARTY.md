# Third-party material in bench/

## pi (pi-coding-agent 0.84.3)

The edit bench drives the model the way the coding agent pi does
([github.com/earendil-works/pi](https://github.com/earendil-works/pi), `packages/coding-agent`):

- `edit_bench.json`: pi's system prompt and its read, edit and write tool definitions, as captured from
  the requests pi 0.84.3 sends (`pi -p --tools read,edit,write`). The install path of pi's docs and the
  working directory are templated (`{pi_dir}`, `{cwd}`).
- `editbench.py`: the tools' result and error texts and the edit tool's matching rules, ported from pi's
  `dist/core/tools` (`read.js`, `truncate.js`, `edit.js`, `edit-diff.js`, `write.js`) and the error texts of
  `@earendil-works/pi-ai` (tool-argument validation) and `@earendil-works/pi-agent-core` (the agent loop).
- `edit_pi_check.mjs` loads pi's own code at run time to check the port; it copies none of it.

pi is under the MIT licence:

```
MIT License

Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
