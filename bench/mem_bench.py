"""Memory benchmark: peak RSS per operation, each in a fresh subprocess.

Usage (per env, like run_bench.py):
  uv run --project bench/baseline  python bench/mem_bench.py --out bench/results/baseline_mem.json
  uv run --project bench/candidate python bench/mem_bench.py --out bench/results/candidate_mem.json

Child mode (internal): python bench/mem_bench.py --child <op>
"""
import argparse
import json
import os
import resource
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

OPS = ["floor", "import", "init", "cut_default", "tokenize", "posseg", "tfidf", "textrank"]


def peak_rss_bytes():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return ru if sys.platform == "darwin" else ru * 1024


def run_child_op(op):
    if op == "floor":
        pass
    else:
        import jieba_rs as jieba

        if op == "import":
            pass
        else:
            jieba.initialize()
            if op != "init":
                long_text = C.load_long_text()
                if op == "cut_default":
                    jieba.lcut(long_text)
                elif op == "tokenize":
                    list(jieba.tokenize(long_text))
                elif op == "posseg":
                    import jieba_rs.posseg as pseg

                    pseg.lcut(long_text)
                elif op == "tfidf":
                    import jieba_rs.analyse

                    jieba.analyse.extract_tags(long_text, topK=20)
                elif op == "textrank":
                    import jieba_rs.analyse

                    jieba.analyse.textrank(long_text, topK=20)
    print(json.dumps({"op": op, "peak_rss_bytes": peak_rss_bytes()}))


def driver(out_path):
    lib = None
    results = {}
    for op in OPS:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", op],
            capture_output=True,
            text=True,
        )
        line = r.stdout.strip().splitlines()[-1]
        data = json.loads(line)
        results[op] = data["peak_rss_bytes"]
        print(f"  {op:14s} peak RSS = {data['peak_rss_bytes']/1e6:8.1f} MB")
    # library identity from a child process
    child_dir = os.path.dirname(os.path.abspath(__file__))
    snippet = (
        "import json,sys;"
        f"sys.path.insert(0, {child_dir!r});"
        "import _common as C;print(json.dumps(C.library_info()))"
    )
    r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    try:
        lib = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        lib = {"library": "?"}

    floor = results.get("floor", 0)
    summary = {
        **lib,
        "machine": C.machine_info(),
        "peak_rss_bytes": results,
        "interpreter_floor_bytes": floor,
        "post_init_delta_bytes": results.get("init", 0) - floor,
        "cut_default_delta_bytes": results.get("cut_default", 0) - results.get("init", 0),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
    print(f"  interpreter floor: {floor/1e6:.1f} MB")
    print(f"  dictionary-resident (post-init - floor): "
          f"{summary['post_init_delta_bytes']/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.child:
        run_child_op(args.child)
    else:
        driver(args.out)


if __name__ == "__main__":
    main()
