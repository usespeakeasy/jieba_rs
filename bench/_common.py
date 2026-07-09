"""Shared helpers for the benchmark scripts (importable in both envs)."""
import hashlib
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "fixtures", "corpus")

# Size of the "long" throughput workload, in characters (~3 bytes each in UTF-8).
LONG_CHARS = 150_000
SHORT_CALLS = 10_000


def library_info():
    import jieba_rs as jieba

    try:
        import jieba_rs._rustjieba  # noqa: F401

        name = "jieba-rs"
    except Exception:
        name = "unknown"
    return {
        "library": name,
        "version": getattr(jieba, "__version__", "?"),
        "module_path": getattr(jieba, "__file__", "?"),
    }


def machine_info():
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    # richer CPU brand on macOS
    try:
        import subprocess

        info["cpu_brand"] = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
    except Exception:
        pass
    return info


def load_long_text():
    with open(os.path.join(CORPUS, "article_zh.txt"), encoding="utf-8") as fh:
        return fh.read()[:LONG_CHARS]


def load_sentences():
    with open(os.path.join(CORPUS, "sentences.txt"), encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def output_hash(items):
    h = hashlib.sha256()
    for it in items:
        h.update(("\x1f".join(it) if isinstance(it, (list, tuple)) else str(it)).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]
