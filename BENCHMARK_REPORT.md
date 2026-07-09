# Benchmark: Production Optimization Pass

This report compares `jieba-rs` before and after the production optimization
pass. The supported contract is the focused public `jieba_rs` API documented in
the README, not complete jieba 0.42.1 implementation parity.

Correctness hashes stayed stable:

- Generic cut guard: `2a8416f7b3b7cf93`

## Environment

- Host: Linux x86_64
- Python: CPython 3.12.13
- Runs: 7 timed runs, median reported
- Local package benchmark command:
  `uv run --project bench/candidate --reinstall-package jieba-rs python bench/run_bench.py`

## Local Package Timings

150k-character long corpus unless noted.

| Operation | Before | After | Change |
|---|--:|--:|--:|
| `cut` precise + HMM | 114.0 ms | 124.1 ms | 0.92x |
| `cut` HMM=False | 78.6 ms | 85.6 ms | 0.92x |
| `cut_all` | 66.6 ms | 68.5 ms | 0.97x |
| `cut_for_search` | 132.6 ms | 119.2 ms | 1.11x |
| `tokenize` | 146.9 ms | 152.4 ms | 0.96x |
| `posseg.cut` | 2264.1 ms | 444.4 ms | **5.09x** |
| `extract_tags` | 151.1 ms | 156.1 ms | 0.97x |
| `textrank` | 2411.9 ms | 637.5 ms | **3.78x** |
| 10k short cuts | 141.3 ms | 149.9 ms | 0.94x |
| 10k `add_word` | 1282.9 ms | 458.3 ms | **2.80x** |
| cold initialize | 1048.4 ms | 444.0 ms | **2.36x** |
| import `jieba_rs` | 120.3 ms | 25.3 ms | **4.75x** |

Several generic segmentation rows are within small-run noise or modestly slower
in this renamed-package run. They should be revisited before making stronger
generic segmentation performance claims.

## Private real-world Chinese corpus

Both libraries produced identical outputs for all 1,633 lines (9,471
characters). Results are medians of 21 measured runs across three interleaved
rounds, each after 2 warmups. Each library and operation ran in an isolated
process. The baseline is jieba-rs commit `0699b8b`.

| Operation | jieba | Baseline jieba-rs | Optimized jieba-rs | Optimized vs jieba |
|---|---:|---:|---:|---:|
| `import` | 107.589 ms | 4.850 ms | 4.766 ms | **22.57x** |
| `initialize` | 201.451 ms | 108.173 ms | 84.597 ms | **2.38x** |
| `cut` | 9.904 ms | 2.646 ms | 2.800 ms | **3.54x** |
| `cut_no_hmm` | 6.927 ms | 2.376 ms | 2.402 ms | **2.88x** |
| `cut_all` | 4.095 ms | 1.753 ms | 1.807 ms | **2.27x** |
| `cut_for_search` | 10.571 ms | 2.875 ms | 2.869 ms | **3.69x** |
| `tokenize` | 10.574 ms | 3.069 ms | 3.034 ms | **3.49x** |
| `posseg` | 87.833 ms | 5.608 ms | 5.602 ms | **15.68x** |

Dictionary initialization is 27.9% faster than the baseline. Steady-state
operation differences are small relative to this short-line workload and move
in both directions; exact outputs match for every compared operation.

Fresh-process median RSS from seven measured processes:

| Phase | jieba | Baseline jieba-rs | Optimized jieba-rs | Saved vs baseline |
|---|---:|---:|---:|---:|
| imported | 25.0 MB | 20.8 MB | 20.8 MB | 0.0 MB |
| dictionary initialized | 98.9 MB | 118.5 MB | 66.6 MB | **51.9 MB** |
| first HMM cut | 98.9 MB | 134.9 MB | 82.8 MB | **52.1 MB** |
| first TF-IDF extraction | 197.7 MB | 180.5 MB | 140.8 MB | **39.7 MB** |

The initialized optimized build uses 43.8% less RSS than the jieba-rs baseline
and 32.3 MB less than jieba 0.42.1. Base dictionary POS tags are interned into
55 shared values and stored in the existing 8-byte frequency entry rather than
in a second 349,046-entry string map.

Machine: Apple M5 Max, Darwin arm64, Python 3.11.15. The corresponding
aggregate-only artifact is `bench/results/private_corpus.json`; it contains no
source text, identifiers, or paths.

## Memory

Peak RSS from fresh subprocesses, normalized to bytes and shown here in MB.

| Measurement | Before | After | Change |
|---|--:|--:|--:|
| import peak | 40.1 MB | 20.1 MB | 49.9% lower |
| initialize peak | 211.7 MB | 137.0 MB | 35.3% lower |
| dictionary-resident delta | 195.4 MB | 120.8 MB | **38.2% lower** |
| `posseg.cut` peak | 285.4 MB | 170.9 MB | **40.1% lower** |
| `textrank` peak | 322.4 MB | 206.5 MB | 36.0% lower |

## What Changed

- POS HMM state/tag keys moved from cloned `String`/`HashMap` values to stable
  integer state IDs, flat transition scores, compact transition adjacency, dense
  score buffers, and integer backpointers.
- Python no longer eagerly parses POS/finalseg HMM JSON tables during import.
- Python no longer materializes a full mutable `Tokenizer.FREQ` mirror during
  initialization; Rust owns dictionary state, while Python keeps a sparse
  mutation overlay with on-demand `FREQ.get` compatibility.
- `suggest_freq(..., tune=True)` delegates to Rust and keeps zero-frequency
  force-split state synchronized.
- `cut_all` now scans dictionary candidates directly instead of building a full
  DAG only to discard most of it.

## Validation

- `cargo test -p jieba-core`
- `uv run pytest`

Native `perf` profiling was attempted but blocked by host policy
(`perf_event_paranoid=4`). Python `cProfile` confirmed the pre-optimization POS
hot path spent nearly all time in the native `RustTokenizer.pos_cut_detailed`
frame, which motivated the Rust-side POS table rewrite.
