# Upstream jieba regression cases

`fixtures/golden/jieba_test_corpus.json` contains the 87 non-empty segmentation
cases from `test/test.py` in jieba v0.42.1, captured against the original Python
implementation. The source tag resolves to commit
`1e20c89b66f56c9301b0feed211733ffaa1bd72a`.

The cases and their original project are MIT licensed. The local test harness
exercises them through `jieba_rs` across precise, no-HMM, full, and search modes.
Additional local tests cover token offsets, POS tagging, dictionary changes,
file-like user dictionaries, concurrency, TF-IDF, TextRank, and POSIX parallel
mode. Optional Paddle and Whoosh integrations are intentionally excluded.

Upstream: https://github.com/fxsjy/jieba/tree/v0.42.1/test
