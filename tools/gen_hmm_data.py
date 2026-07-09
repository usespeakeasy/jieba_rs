"""Dump jieba's HMM probability tables to a single JSON consumed by the Rust core.

Run:  uv run --no-project python tools/gen_hmm_data.py

Reads the vendored reference tables under reference/jieba/ and writes
python/jieba_rs/data/hmm_data.json.

Encoding of posseg (state, tag) tuple keys: a single JSON string ``state + tag``
where ``state`` is one char of BMES and ``tag`` is the (lowercase, may be
multi-char) POS tag, e.g. ('B','nr') -> "Bnr". Decoding in Rust: first char is
state, remainder is tag.
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference", "jieba", "jieba")
OUT = os.path.join(ROOT, "python", "jieba_rs", "data", "hmm_data.json")


def load_P(path):
    """Import a jieba prob_*.py module and return its ``P`` attribute."""
    spec = importlib.util.spec_from_file_location("_p", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.P


def enc_key(k):
    """Encode a (state, tag) tuple as 'state'+'tag'; pass through plain str."""
    if isinstance(k, tuple):
        state, tag = k
        return state + tag
    return k


def enc_map(d):
    return {enc_key(k): v for k, v in d.items()}


def enc_nested(d):
    return {enc_key(k): enc_map(v) for k, v in d.items()}


def main():
    # ---- finalseg (states are plain 'B'/'M'/'E'/'S') ----
    fs_start = load_P(os.path.join(REF, "finalseg", "prob_start.py"))
    fs_trans = load_P(os.path.join(REF, "finalseg", "prob_trans.py"))
    fs_emit = load_P(os.path.join(REF, "finalseg", "prob_emit.py"))

    finalseg = {
        "start": dict(fs_start),
        "trans": {k: dict(v) for k, v in fs_trans.items()},
        "emit": {k: dict(v) for k, v in fs_emit.items()},
    }

    # ---- posseg (keys are (state, tag) tuples) ----
    ps_start = load_P(os.path.join(REF, "posseg", "prob_start.py"))
    ps_trans = load_P(os.path.join(REF, "posseg", "prob_trans.py"))
    ps_emit = load_P(os.path.join(REF, "posseg", "prob_emit.py"))
    ps_cst = load_P(os.path.join(REF, "posseg", "char_state_tab.py"))

    posseg = {
        "start": enc_map(ps_start),
        "trans": enc_nested(ps_trans),
        "emit": enc_nested(ps_emit),
        # char -> list of encoded (state,tag) keys, order preserved
        "char_state_tab": {ch: [enc_key(k) for k in states]
                           for ch, states in ps_cst.items()},
    }

    data = {"finalseg": finalseg, "posseg": posseg}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, sort_keys=True)
        fh.write("\n")

    size = os.path.getsize(OUT)
    print("wrote %s (%.1f MB)" % (OUT, size / 1e6))
    print("finalseg states:", sorted(finalseg["start"].keys()))
    print("posseg start entries:", len(posseg["start"]))
    print("posseg char_state_tab entries:", len(posseg["char_state_tab"]))


if __name__ == "__main__":
    sys.exit(main())
