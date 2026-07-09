# Benchmark & Profiling Protocol: `jieba` (Python) vs `jieba-rs`

Compares the original `jieba==0.42.1` (baseline) against `jieba-rs` imported
as `jieba_rs` (candidate) on wall time, throughput, CPU usage, and memory.
Everything runs through `uv`; the baseline and candidate live in separate uv
projects.

## 0. Preconditions

- Parity suite green (`uv run pytest`) — never benchmark a library that produces different output; speed comparisons are meaningless otherwise. The harness re-verifies this (step 3).
- Built release wheel in `dist/` (`uv build`).
- Quiet machine: on AC power, no browsers/builds running. Note hardware in the report (`sysctl -n machdep.cpu.brand_string`, RAM, OS, Python version).

## 1. Environments

```
bench/baseline/pyproject.toml   → [project] dependencies = ["jieba==0.42.1", "psutil"]
bench/candidate/pyproject.toml  → dependencies = ["psutil"] + jieba-rs wheel:
    [tool.uv.sources] jieba-rs = { path = "../../dist/<wheel>", }   # or pip-style direct path dep
```
Run everything as `uv run --project bench/baseline ...` vs `uv run --project bench/candidate ...`. The bench scripts live in `bench/` (shared) and take `--out <json>`.

## 2. Workloads

From `fixtures/corpus/`:
- **long**: `article_zh.txt` (≥ 500 KB continuous Chinese prose) — throughput workload.
- **short**: `sentences.txt` lines cycled to 10,000 calls — per-call overhead workload (crossing the FFI boundary per call is where Rust bindings often lose; measure it honestly).
- **mixed**: a 100 KB slice of CJK + ASCII + numbers + whitespace.

Operations to time (each on the workloads that make sense):

| id | call |
|----|------|
| init | subprocess: `import jieba_rs as jieba; jieba.initialize()` |
| cut_default | `list(jieba.cut(text))` |
| cut_no_hmm | `list(jieba.cut(text, HMM=False))` |
| cut_all | `list(jieba.cut(text, cut_all=True))` |
| cut_search | `list(jieba.cut_for_search(text))` |
| tokenize | `list(jieba.tokenize(text))` |
| posseg | `list(jieba_rs.posseg.cut(text))` |
| tfidf | `jieba.analyse.extract_tags(long_text, topK=20)` |
| textrank | `jieba.analyse.textrank(long_text, topK=20)` |
| add_word | 10,000 `add_word` calls (fresh Tokenizer) |
| short_calls | 10,000 × `list(jieba.cut(line))` over sentences |

## 3. Timing methodology (`bench/run_bench.py`)

- Correctness guard first: hash the `cut_default` output of the first 100 sentences and store it in the JSON; the report step fails if baseline and candidate hashes differ.
- Per operation: 2 warmup runs, then **N ≥ 7 timed runs** of `time.perf_counter()` around the call. Record all runs; report median, min, p90.
- CPU: wrap each run with `os.times()`; report `(utime+stime) / wall` as utilization (catches "faster wall but burns threads" effects).
- Throughput: `len(text) / median_seconds` → chars/s; also MB/s (UTF-8 bytes).
- **Cold init** is special: measure in a fresh subprocess per run (`subprocess.run([sys.executable, "-c", "import time; import jieba_rs as jieba; t=time.perf_counter(); jieba.initialize(); print(time.perf_counter()-t)"])`), 7 runs. Also measure bare `import jieba_rs` time the same way. Optionally cross-check with `hyperfine` if installed.
- Output JSON schema: `{ "library": ..., "version": ..., "machine": {...}, "results": { op_id: { "workload": ..., "runs_s": [...], "median_s": ..., "cpu_util": ..., "chars_per_s": ..., "output_hash": ... } } }` → `bench/results/{baseline,candidate}.json`.

## 4. Memory profiling (`bench/mem_bench.py`)

Two measurements, both per-operation, both in **fresh subprocesses** so numbers don't contaminate each other:

1. **Peak RSS**: child process runs the op once, then reports `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`. ⚠️ macOS reports **bytes**, Linux **KiB** — normalize via `sys.platform`. Also record baseline peak of an empty child (interpreter floor) and report the delta.
2. **RSS timeline**: parent samples child `psutil.Process(pid).memory_info().rss` every 50 ms during the op; store the series. This shows steady-state after-init footprint vs transient segmentation spikes.

Report at minimum: interpreter floor, post-`initialize()` RSS (dictionary-resident footprint — the headline memory number), peak RSS during `cut_default` on the long text, and peak during `textrank`.

Optional deep dive: `memray run --native` on the baseline (Python allocations) — note that Rust-side allocations are invisible to Python tracers; the subprocess RSS numbers are the fair cross-library comparison. Do not use `tracemalloc` for cross-library claims.

## 5. CPU profiling (flamegraphs)

- Baseline: `uv run --project bench/baseline py-spy record -o bench/results/baseline_cut.svg -- python bench/profile_target.py cut_default` (`uv tool install py-spy` if needed; on macOS py-spy may need `sudo`).
- Candidate Python-side: same py-spy run — expect the native call to dominate,
  plus generator-resume frames required for exact `types.GeneratorType` behavior.
- Candidate Rust-side (optional): `cargo bench` in `crates/jieba-core` with a `criterion` bench over the same article, and/or `samply record` on a Rust binary target, to find hot spots (expect: HashMap lookups in `get_DAG`, allocation in word yielding).

## 6. Report (`bench/report.py` → `BENCHMARK_REPORT.md`)

One row per operation:

| Operation | Workload | jieba median | jieba-rs median | Speedup | jieba chars/s | jieba-rs chars/s | jieba peak RSS | jieba-rs peak RSS |

Plus sections: machine info; init/import times; memory footprint summary; CPU utilization notes; flamegraph links; methodology caveats (single machine, macOS RSS semantics, FFI-boundary note for `short_calls`); and the correctness-hash confirmation line.

State results honestly: if the candidate loses any row (plausible for `short_calls` or `add_word` due to per-call FFI + string conversion costs), report it and open an optimization note — do not omit losing rows.
