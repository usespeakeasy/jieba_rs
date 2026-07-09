"""A long-running target for CPU profilers (py-spy / samply).

Usage: python bench/profile_target.py <op> [seconds]
Repeats the op on the long text for ~`seconds` so a sampler can collect frames.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def main():
    op = sys.argv[1] if len(sys.argv) > 1 else "cut_default"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    import jieba_rs as jieba

    jieba.initialize()
    text = C.load_long_text()

    import jieba_rs.analyse  # noqa: F401
    import jieba_rs.posseg as pseg  # noqa: F401

    def run():
        if op == "cut_default":
            jieba.lcut(text)
        elif op == "posseg":
            pseg.lcut(text)
        elif op == "tfidf":
            jieba.analyse.extract_tags(text, topK=20)
        else:
            jieba.lcut(text)

    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        run()
        n += 1
    sys.stderr.write("ran %d iterations of %s\n" % (n, op))


if __name__ == "__main__":
    main()
