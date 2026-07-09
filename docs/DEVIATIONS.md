# Supported Contract and Deviations

The current compatibility target is the focused public `jieba_rs` API, not
complete Python jieba 0.42.1 drop-in parity. The supported surface is:

- `import jieba_rs`
- `import jieba_rs.posseg as pseg`
- `jieba_rs.initialize()`
- `jieba_rs.dt.tmp_dir` assignment
- `jieba_rs.add_word`, `jieba_rs.del_word`, `jieba_rs.suggest_freq`
- `jieba_rs.cut(text, cut_all=True)`
- `pseg.cut(text)`
- `pseg.pair(word, flag)`
- POS items exposing `.word` and `.flag`

Segmentation and POS outputs are guarded by local golden and behavior tests.

## Removed optional integrations

Paddle/LAC segmentation and the Whoosh analyzer are not part of the supported
package. `enable_paddle`, the `use_paddle` keyword, `jieba_rs.lac_small`, and
`jieba_rs.analyse.ChineseAnalyzer` are absent. Applications that need those
optional jieba 0.42.1 integrations should continue using the original package.

## Intentional deviations

- `Tokenizer.FREQ` is a sparse Python mutation overlay. `FREQ.get("word")` can
  still read initialized Rust dictionary frequencies on demand, but the full
  base dictionary is no longer materialized as a mutable Python dict.
- The marshal dictionary cache is no longer used during initialization.
  `tmp_dir` and `cache_file` remain assignable compatibility attributes.
- POS dictionary tags come from Rust's dictionary/tag state. Mutating
  `POSTokenizer.word_tag_tab` is not part of the supported contract.
- Python POS/finalseg HMM model tables are loaded lazily only for legacy helper
  calls; native segmentation and POS do not use Python model dictionaries.
- `finalseg.Force_Split_Words` remains process-global and is synchronized into
  all live Rust tokenizers, including direct set mutations.

See `BENCHMARK_REPORT.md` for current performance and memory numbers.
