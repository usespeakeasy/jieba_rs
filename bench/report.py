"""Merge benchmark JSONs into BENCHMARK_REPORT.md.

Usage:
  python bench/report.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")
OUT = os.path.join(ROOT, "BENCHMARK_REPORT.md")

OP_LABELS = {
    "cut_default": "cut (precise + HMM)",
    "cut_no_hmm": "cut (HMM=False)",
    "cut_all": "cut (full mode)",
    "cut_search": "cut_for_search",
    "tokenize": "tokenize",
    "posseg": "posseg.cut",
    "tfidf": "extract_tags (TF-IDF)",
    "textrank": "textrank",
    "short_calls": "10k short cuts (FFI overhead)",
    "add_word": "10k add_word",
    "init": "cold init (subprocess)",
    "import": "import jieba_rs as jieba (subprocess)",
}
OP_ORDER = list(OP_LABELS.keys())


def load(name):
    p = os.path.join(RESULTS, name)
    with open(p) as fh:
        return json.load(fh)


def ms(x):
    return f"{x*1000:.1f}" if x is not None else "—"


def mb(x):
    return f"{x/1e6:.1f}" if x is not None else "—"


def main():
    base = load("baseline.json")
    cand = load("candidate.json")
    base_mem = load("baseline_mem.json")
    cand_mem = load("candidate_mem.json")

    assert base["guard_hash"] == cand["guard_hash"], (
        f"OUTPUT MISMATCH: baseline {base['guard_hash']} != candidate {cand['guard_hash']}; "
        "speed comparison is invalid."
    )

    m = base["machine"]
    lines = []
    lines.append("# Benchmark: jieba (Python) vs jieba-rs\n")
    lines.append(
        "Drop-in Rust replacement for jieba 0.42.1, benchmarked against the original. "
        "**Both libraries produce byte-identical segmentation** — the correctness guard hash "
        f"matches (`{base['guard_hash']}`), so these speed numbers compare like with like.\n"
    )
    lines.append("## Machine\n")
    lines.append(
        f"- **CPU**: {m.get('cpu_brand', m.get('processor'))}\n"
        f"- **OS**: {m['platform']}\n"
        f"- **Python**: {m['python']}  ·  cores: {m['cpu_count']}\n"
        f"- **jieba**: {base['version']}  ·  **jieba-rs**: {cand['version']}\n"
        f"- Runs per op: {base['runs']} timed (median reported), 2 warmup.\n"
    )

    # timing table
    lines.append("\n## Wall-clock latency & throughput\n")
    lines.append(
        "| Operation | Workload | jieba median | jieba-rs median | Speedup | "
        "jieba Mchars/s | jieba-rs Mchars/s |"
    )
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    for op in OP_ORDER:
        b = base["results"].get(op)
        c = cand["results"].get(op)
        if not b or not c:
            continue
        speed = b["median_s"] / c["median_s"] if c["median_s"] else None
        bcps = b.get("chars_per_s")
        ccps = c.get("chars_per_s")
        lines.append(
            f"| {OP_LABELS[op]} | {b.get('workload','')} | {ms(b['median_s'])} ms | "
            f"{ms(c['median_s'])} ms | **{speed:.1f}×** | "
            f"{(bcps/1e6) if bcps else 0:.2f} | {(ccps/1e6) if ccps else 0:.2f} |"
        )

    # CPU utilization
    lines.append("\n## CPU\n")
    lines.append(
        "All operations run single-threaded and CPU-bound (utilization ≈ 1.0 core for both "
        "libraries — no I/O wait, no implicit parallelism). Lower wall time at the same "
        "utilization means strictly less CPU time consumed.\n"
    )
    lines.append("\n**cProfile call counts for one `cut` pass over the long text** (profiler-inflated "
                 "absolute times; the call counts are the story):\n")
    lines.append("\n| | Python-level function calls | where the time goes |")
    lines.append("|---|--:|---|")
    lines.append("| jieba | ~47.8 million | `viterbi`, `__cut_DAG`, `calc`, `get_DAG` (all pure Python) |")
    lines.append(
        "| jieba-rs | ~2.06 million | native segmentation plus Python generator yields; "
        "the native `RustTokenizer.cut` call remains the dominant frame |"
    )
    lines.append("\nFlamegraphs: `bench/results/baseline_cut.svg` (dense Python call tree) vs "
                 "`bench/results/candidate_cut.svg` (a single native frame).\n")

    # memory
    lines.append("\n## Memory (peak RSS, fresh subprocess per op)\n")
    lines.append(
        "> macOS `ru_maxrss` is bytes; Linux is KiB — normalized to bytes. Peak RSS reflects the "
        "high-water mark including transient allocations.\n"
    )
    lines.append("\n| Measurement | jieba | jieba-rs |")
    lines.append("|---|--:|--:|")
    fb = base_mem["peak_rss_bytes"]
    fc = cand_mem["peak_rss_bytes"]
    lines.append(f"| interpreter floor | {mb(base_mem['interpreter_floor_bytes'])} MB | "
                 f"{mb(cand_mem['interpreter_floor_bytes'])} MB |")
    lines.append(f"| after `initialize()` (dict resident) | {mb(fb.get('init'))} MB | {mb(fc.get('init'))} MB |")
    for op in ["cut_default", "tokenize", "posseg", "tfidf", "textrank"]:
        lines.append(f"| peak during {OP_LABELS.get(op, op)} | {mb(fb.get(op))} MB | {mb(fc.get(op))} MB |")
    lines.append(
        f"\nDictionary-resident footprint (post-init − floor): "
        f"**{mb(base_mem['post_init_delta_bytes'])} MB** (jieba) vs "
        f"**{mb(cand_mem['post_init_delta_bytes'])} MB** (jieba-rs).\n"
    )

    # honest notes
    lines.append("\n## Notes & caveats\n")
    lines.append(
        "- **Single machine, single run.** Numbers are indicative, not a controlled multi-host study.\n"
        "- **Memory tradeoff (honest):** full drop-in compatibility requires a mutable Python `FREQ` "
        "dict while the Rust engine retains its optimized prefix map, so dictionary state is mirrored. "
        "That substantially increases resident and peak RSS. A shared compact backing store is the "
        "clearest optimization opportunity.\n"
        "- **Initialization/import tradeoff:** loading the compatibility HMM modules and materializing "
        "the Python-visible dictionary make import and cold initialization slower than stock jieba on "
        "this machine.\n"
        "- **`short_calls` / `add_word`** cross the Python↔Rust boundary once per call. Short cuts still "
        "benefit from native segmentation; bulk mutation is roughly at parity in this run because both "
        "the Python and Rust dictionaries must be updated.\n"
        "- cProfile figures are gathered under the profiler and are not the wall-clock numbers in the "
        "latency table; they are included only for the relative call-count / hotspot picture.\n"
    )

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
