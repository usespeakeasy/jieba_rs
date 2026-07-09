"""Timing benchmark: run against whichever `jieba_rs` is installed in the env.

Usage:
  uv run --project bench/baseline  python bench/run_bench.py --out bench/results/baseline.json
  uv run --project bench/candidate python bench/run_bench.py --out bench/results/candidate.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

WARMUP = 2
RUNS = 7


def timed(fn, runs=RUNS):
    """Return (list of wall times, cpu_util) for `fn`. 2 warmups first."""
    for _ in range(WARMUP):
        fn()
    times = []
    t_cpu0 = os.times()
    w0 = time.perf_counter()
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        times.append(time.perf_counter() - s)
    wall = time.perf_counter() - w0
    t_cpu1 = os.times()
    cpu = (t_cpu1.user - t_cpu0.user) + (t_cpu1.system - t_cpu0.system)
    util = cpu / wall if wall > 0 else 0.0
    return times, util


def stats(times):
    s = sorted(times)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    p90 = s[min(n - 1, int(round(0.9 * (n - 1))))]
    return {"runs_s": times, "median_s": median, "min_s": s[0], "p90_s": p90}


def cold_init_times(runs=RUNS):
    """Measure import+initialize in fresh subprocesses."""
    code_init = (
        "import time;import jieba_rs as jieba;"
        "t=time.perf_counter();jieba.initialize();"
        "print(time.perf_counter()-t)"
    )
    code_import = "import time;t=time.perf_counter();import jieba_rs as jieba;print(time.perf_counter()-t)"
    inits, imports = [], []
    for _ in range(runs):
        r = subprocess.run([sys.executable, "-c", code_init], capture_output=True, text=True)
        inits.append(float(r.stdout.strip()))
        r = subprocess.run([sys.executable, "-c", code_import], capture_output=True, text=True)
        imports.append(float(r.stdout.strip()))
    return stats(inits), stats(imports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=RUNS)
    args = ap.parse_args()

    import jieba_rs as jieba
    import jieba_rs.analyse
    import jieba_rs.posseg as pseg

    jieba.initialize()
    long_text = C.load_long_text()
    sentences = C.load_sentences()
    n_long = len(long_text)

    results = {}

    def record(op, fn, workload, size_chars, extra=None):
        times, util = timed(fn, args.runs)
        st = stats(times)
        st["cpu_util"] = util
        st["workload"] = workload
        st["chars"] = size_chars
        st["chars_per_s"] = size_chars / st["median_s"] if st["median_s"] else None
        if extra:
            st.update(extra)
        results[op] = st
        print(f"  {op:16s} median={st['median_s']*1000:8.2f} ms  "
              f"{(st['chars_per_s'] or 0)/1e6:6.2f} Mchars/s  cpu_util={util:.2f}")

    print(f"[{C.library_info()['library']}] timing on {n_long} chars long / "
          f"{len(sentences)} sentences")

    # correctness guard: hash cut_default of first 100 sentences
    guard = C.output_hash([jieba.lcut(s) for s in sentences[:100]])

    record("cut_default", lambda: jieba.lcut(long_text, HMM=True), "long", n_long)
    record("cut_no_hmm", lambda: jieba.lcut(long_text, HMM=False), "long", n_long)
    record("cut_all", lambda: jieba.lcut(long_text, cut_all=True), "long", n_long)
    record("cut_search", lambda: jieba.lcut_for_search(long_text), "long", n_long)
    record("tokenize", lambda: list(jieba.tokenize(long_text)), "long", n_long)
    record("posseg", lambda: pseg.lcut(long_text), "long", n_long)
    record("tfidf", lambda: jieba.analyse.extract_tags(long_text, topK=20), "long", n_long)
    record("textrank", lambda: jieba.analyse.textrank(long_text, topK=20), "long", n_long)

    total_short_chars = sum(len(s) for s in sentences) * (C.SHORT_CALLS // len(sentences) + 1)
    short_slice = (sentences * (C.SHORT_CALLS // len(sentences) + 1))[: C.SHORT_CALLS]

    def short_calls():
        for s in short_slice:
            jieba.lcut(s)

    record("short_calls", short_calls, "short", sum(len(s) for s in short_slice),
           extra={"n_calls": C.SHORT_CALLS})

    def add_word_bulk():
        t = jieba.Tokenizer()
        t.initialize()
        for i in range(10000):
            t.add_word("测试词%d" % i, 100)

    record("add_word", add_word_bulk, "n/a", 0, extra={"n_calls": 10000})

    print("  measuring cold init (subprocess)...")
    init_stats, import_stats = cold_init_times(args.runs)
    results["init"] = {**init_stats, "workload": "subprocess"}
    results["import"] = {**import_stats, "workload": "subprocess"}
    print(f"  init median={init_stats['median_s']*1000:.1f} ms  "
          f"import median={import_stats['median_s']*1000:.1f} ms")

    out = {
        **C.library_info(),
        "machine": C.machine_info(),
        "runs": args.runs,
        "guard_hash": guard,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}  (guard={guard})")


if __name__ == "__main__":
    main()
