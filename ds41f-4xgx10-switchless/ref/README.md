# ref/

Reference material the engine is checked against.

- DeepSeek's, copied from the DeepSeek-V4.1-Flash release (MIT, Copyright (c) 2023 DeepSeek,
  `LICENSE-deepseek`): `inference/` (model.py, kernel.py, convert.py, generate.py, engram.py, ...),
  `encoding/` (the chat encoding and its tests), `config.json`, `tokenizer_config.json`, and
  `techreport.txt`, the tech report's text. `encoding-1017/` is the same directory from the
  DeepSeek-V4.1-Flash-1017 release (revision 1bae0e10), whose chat encoding `dsfront --encoding 1017`
  follows.
- This project's: `torchref/` (a pure-torch kernel.py and a layer-streaming driver around model.py, for
  the golden data), `inventory.json` and `inventory.txt` (every checkpoint tensor, aggregated by layer,
  from `tools/st_inventory.py`), `ext/`.

The root `NOTICE` lists every third-party licence in the repository.
