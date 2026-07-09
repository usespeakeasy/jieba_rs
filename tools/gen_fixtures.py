#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate deterministic golden-output fixtures from the REAL Python jieba.

Run with an isolated jieba 0.42.1:

    uv run --with jieba==0.42.1 --no-project python tools/gen_fixtures.py

Outputs (all under fixtures/golden/):
    segmentation.json      per-sentence cut/tokenize/posseg outputs
    analyse.json           extract_tags + textrank on the full article
    mutation.json          dictionary-mutation scenarios
    jieba_test_corpus.json cuts of the reference test.py sentence list

Determinism: jieba is deterministic, so repeated runs produce identical
files. The only global-state mutation (module-level suggest_freq on
jieba.dt) is performed LAST, after every pristine-dictionary fixture has
already been written, so ordering within a run never leaks state.
"""

import json
import os
import re
import sys

import jieba
import jieba.analyse
import jieba.posseg as pseg

# --- paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "fixtures", "corpus")
GOLDEN = os.path.join(ROOT, "fixtures", "golden")
SENTENCES_TXT = os.path.join(CORPUS, "sentences.txt")
ARTICLE_TXT = os.path.join(CORPUS, "article_zh.txt")
USERDICT_TXT = os.path.join(CORPUS, "userdict.txt")

os.makedirs(GOLDEN, exist_ok=True)


def dump(obj, name):
    """Write obj as pretty, deterministic JSON with a trailing newline."""
    path = os.path.join(GOLDEN, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=True, indent=1)
        fh.write("\n")
    print("wrote %s (%d bytes)" % (name, os.path.getsize(path)))


def read_lines(path, limit=None):
    """Read a corpus file, keeping each line's content exactly (only the
    trailing newline is stripped), skipping only truly empty lines."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = raw.rstrip("\n")
            if line == "":
                continue
            out.append(line)
    return out


# --- sentence set ----------------------------------------------------------
def build_sentence_set():
    sset = read_lines(SENTENCES_TXT)
    sset += read_lines(ARTICLE_TXT, limit=2000)
    return sset


def seg_record(s, search_and_pos=True):
    rec = {
        "s": s,
        "cut_default": jieba.lcut(s),
        "cut_no_hmm": jieba.lcut(s, HMM=False),
        "cut_all": jieba.lcut(s, cut_all=True),
        "cut_for_search": jieba.lcut_for_search(s),
    }
    if search_and_pos:
        rec["cut_for_search_no_hmm"] = jieba.lcut_for_search(s, HMM=False)
        rec["tokenize_default"] = [
            [w, start, end] for (w, start, end) in jieba.tokenize(s)
        ]
        rec["tokenize_search"] = [
            [w, start, end] for (w, start, end) in jieba.tokenize(s, mode="search")
        ]
        rec["posseg"] = [[p.word, p.flag] for p in pseg.cut(s)]
        rec["posseg_no_hmm"] = [[p.word, p.flag] for p in pseg.cut(s, HMM=False)]
    return rec


def gen_segmentation():
    sset = build_sentence_set()
    records = [seg_record(s) for s in sset]
    dump(records, "segmentation.json")
    print("  segmentation: %d sentences" % len(records))


# --- analyse ---------------------------------------------------------------
def gen_analyse():
    with open(ARTICLE_TXT, encoding="utf-8") as fh:
        article = fh.read()
    obj = {
        "extract_tags_top50": [
            [w, wt]
            for (w, wt) in jieba.analyse.extract_tags(
                article, topK=50, withWeight=True
            )
        ],
        "extract_tags_top50_pos": [
            [w, wt]
            for (w, wt) in jieba.analyse.extract_tags(
                article, topK=50, withWeight=True, allowPOS=("ns", "n", "vn", "v")
            )
        ],
        "textrank_top50": [
            [w, wt]
            for (w, wt) in jieba.analyse.textrank(article, topK=50, withWeight=True)
        ],
    }
    dump(obj, "analyse.json")


# --- mutation --------------------------------------------------------------
PROBE_WORDS = [
    "云计算", "韩玉赏鉴", "台中", "石墨烯", "创新办", "凱特琳", "自定义词",
    "中", "将", "北京", "清华大学", "中国", "科学院", "计算所", "李小福",
    "网易", "杭研", "大厦", "叛徒", "不存在的词xyz",
]

SUGGEST_SEGMENTS = [
    "台中",
    ("中", "将"),
    "石墨烯",
    ("今天", "天气"),
    "清华大学",
    ("北京", "大学"),
    "中国科学院",
    "a",
    ("结婚", "的", "和尚"),
    "结婚",
]


def seg_repr(seg):
    return seg if isinstance(seg, str) else "|".join(seg)


def gen_mutation():
    obj = {}

    # --- fresh-Tokenizer before/after probes -------------------------------
    t = jieba.Tokenizer()
    t.initialize()
    obj["initial_total"] = t.total

    t.add_word("石墨烯")
    obj["after_add_shimoxi_total"] = t.total
    obj["after_add_shimoxi_freq"] = t.FREQ.get("石墨烯")

    t.add_word("凱特琳")
    obj["after_add_kailin_total"] = t.total
    obj["after_add_kailin_freq"] = t.FREQ.get("凱特琳")

    t.del_word("自定义词")
    obj["after_del_zidingyi_total"] = t.total
    obj["after_del_zidingyi_freq"] = t.FREQ.get("自定义词")

    t.load_userdict(USERDICT_TXT)
    obj["after_load_userdict_total"] = t.total

    obj["probe_freq_after_userdict"] = {w: t.FREQ.get(w, None) for w in PROBE_WORDS}

    obj["mutated_cut_lisiaofu"] = t.lcut("李小福是创新办主任也是云计算方面的专家")
    obj["mutated_cut_shimoxi"] = t.lcut("石墨烯是一种材料")
    obj["mutated_cut_hanyu"] = t.lcut("韩玉赏鉴很好")

    # --- suggest_freq(tune=False) samples on a separate fresh tokenizer ----
    t2 = jieba.Tokenizer()
    t2.initialize()
    obj["suggest_freq_samples"] = [
        [seg_repr(seg), t2.suggest_freq(seg, tune=False)] for seg in SUGGEST_SEGMENTS
    ]

    # --- module-level suggest_freq (mutates jieba.dt) --- do this LAST ------
    ret = jieba.suggest_freq(("中", "将"), tune=True)
    obj["suggest_freq_zhong_jiang"] = {
        "return": ret,
        "cut_no_hmm": jieba.lcut("如果放到post中将出错。", HMM=False),
    }

    dump(obj, "mutation.json")


# --- reference test corpus -------------------------------------------------
def gen_test_corpus():
    test_py = os.path.join(ROOT, "reference", "jieba", "test", "test.py")
    sentences = []
    if os.path.exists(test_py):
        with open(test_py, encoding="utf-8") as fh:
            src = fh.read()
        # Extract the argument of every cuttest("...") / cuttest('...') call.
        for m in re.finditer(r"""cuttest\(\s*(['"])(.*?)\1\s*\)""", src, re.S):
            sentences.append(m.group(2))
    records = []
    for s in sentences:
        if s == "":
            continue
        records.append(seg_record(s, search_and_pos=False))
    dump(records, "jieba_test_corpus.json")
    print("  jieba_test_corpus: %d sentences" % len(records))


def main():
    # Order matters: every fixture that relies on a pristine module-level
    # jieba.dt runs before gen_mutation()'s module-level suggest_freq.
    gen_segmentation()
    gen_test_corpus()
    gen_analyse()
    gen_mutation()
    print("done: jieba %s" % jieba.__version__)


if __name__ == "__main__":
    main()
