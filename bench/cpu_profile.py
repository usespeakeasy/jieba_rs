"""CPU profiling via cProfile (root-free; portable alternative to py-spy).

For the baseline this surfaces the Python hotspots (__cut_DAG, calc, get_DAG,
viterbi). For the candidate it shows the native RustTokenizer.cut call plus the
generator-resume frames required by exact Python API compatibility.

Usage (per env):
  uv run --project bench/baseline  python bench/cpu_profile.py --op cut_default --out bench/results/baseline_cpu.txt
  uv run --project bench/candidate python bench/cpu_profile.py --op cut_default --out bench/results/candidate_cpu.txt
"""
import argparse
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="cut_default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import jieba_rs as jieba
    import jieba_rs.posseg as pseg

    jieba.initialize()
    text = C.load_long_text()

    def run():
        for _ in range(args.iters):
            if args.op == "posseg":
                pseg.lcut(text)
            else:
                jieba.lcut(text)

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    lib = C.library_info()
    s.write(f"# {lib['library']} {lib['version']} — cProfile op={args.op} iters={args.iters}\n")
    s.write(f"# machine: {C.machine_info().get('cpu_brand', C.machine_info()['processor'])}\n\n")
    ps.print_stats(25)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(s.getvalue())
    # also dump raw stats for flamegraph generation (flameprof)
    prof_path = os.path.splitext(args.out)[0] + ".prof"
    pr.dump_stats(prof_path)
    print(f"wrote {args.out}")
    # echo the top few lines
    print("\n".join(s.getvalue().splitlines()[:16]))


if __name__ == "__main__":
    main()
