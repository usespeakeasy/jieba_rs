# jieba-rs

A Rust-powered Chinese word-segmentation library based on the Python
[`jieba`](https://github.com/fxsjy/jieba) Chinese word-segmentation library.
Rust core + PyO3 bindings, managed with `uv`, Python **3.11+**.

The distribution name is `jieba-rs`; the Python import package is `jieba_rs`.
The legacy `jieba` import is not provided.

```bash
pip uninstall jieba
pip install jieba-rs
# or, with uv:  uv add jieba-rs
```

```python
import jieba_rs
jieba_rs.lcut("我来到北京清华大学")          # ['我', '来到', '北京', '清华大学']
list(jieba_rs.cut("小明硕士毕业于中国科学院计算所"))

import jieba_rs.posseg as pseg
[(p.word, p.flag) for p in pseg.cut("我爱北京天安门")]   # [('我','r'),('爱','v'),('北京','ns'),('天安门','ns')]

import jieba_rs.analyse
jieba_rs.analyse.extract_tags(text, topK=20, withWeight=True)
jieba_rs.analyse.textrank(text, topK=20)
```

## Compatibility

This package exposes a focused Python API under `jieba_rs`:
`jieba_rs.initialize()`, `add_word`, `del_word`, `suggest_freq`,
`jieba_rs.cut(..., cut_all=True)`, `jieba_rs.posseg.cut`, `pseg.pair`, and POS
results with `.word` / `.flag`.

It is no longer intended to be a complete byte-for-byte drop-in replacement for
every jieba 0.42.1 API or implementation detail. In particular, the Python
`Tokenizer.FREQ` table is now a sparse mutation overlay with on-demand Rust
lookups instead of a fully materialized 400k-entry public dict, Python HMM model
tables are lazy compatibility data, and Rust's dictionary/tag state is
authoritative for segmentation and POS.

The optional Paddle/LAC and Whoosh integrations from jieba 0.42.1 are not
included. `enable_paddle`, the `use_paddle` keyword, the `lac_small` namespace,
and the Whoosh `ChineseAnalyzer` adapter are intentionally unsupported. Core
segmentation, POS tagging, dictionary mutation, search tokenization, keyword
analysis, and POSIX parallel mode remain compatibility targets.

## Performance

Current optimization results compare this checkout before/after the production
optimization pass on Linux x86_64. Local correctness guard hashes are unchanged.

| Workload | Before | After | Change |
|---|--:|--:|--:|
| `posseg.cut` long corpus | 2264 ms | 444 ms | **5.1x faster** |
| cold `jieba_rs.initialize()` | 1048 ms | 444 ms | **2.4x faster** |
| `import jieba_rs` | 120 ms | 25.3 ms | **4.8x faster** |
| 10k `add_word` setup | 1283 ms | 458 ms | **2.8x faster** |
| initialized RSS delta | 195 MB | 121 MB | **38% lower** |

Compact, interned dictionary POS tags subsequently reduced initialized RSS on
Apple M5 Max from 118.5 MB to 66.6 MB and made initialization 27.9% faster,
while preserving exact output parity. The current three-way results are in the
benchmark report.

Full benchmark tables, memory numbers, hashes, and known remaining regressions
are in `BENCHMARK_REPORT.md`, including an aggregate-only comparison against
jieba 0.42.1 over a private real-world Chinese corpus.

## Development

```bash
uv sync
uv run maturin develop --release     # build the Rust extension into the venv
cargo test -p jieba-core             # Rust unit tests
uv run pytest                        # Python parity/behavior tests
uv build                             # build the abi3 wheel + sdist
```

Layout: `crates/jieba-core` (pure Rust algorithms), `crates/jieba-py` (PyO3
bindings), `python/jieba_rs` (the Python package). The supported contract and
deviations are written up in `docs/DEVIATIONS.md`.

## Credits & license

Algorithms and bundled data files (`dict.txt`, `idf.txt`, HMM tables) come from
[fxsjy/jieba](https://github.com/fxsjy/jieba), MIT licensed. This project is also
MIT licensed.
