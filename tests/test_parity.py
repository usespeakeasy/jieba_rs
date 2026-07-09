"""T4.1 — byte-exact parity against golden outputs captured from real jieba 0.42.1."""
import jieba_rs as jieba
import jieba_rs.analyse
import jieba_rs.posseg as pseg
import pytest


def _seg_cases(seg):
    return seg


def test_cut_modes_parity(segmentation):
    mismatches = []
    for i, case in enumerate(segmentation):
        s = case["s"]
        checks = {
            "cut_default": jieba.lcut(s),
            "cut_no_hmm": jieba.lcut(s, HMM=False),
            "cut_all": jieba.lcut(s, cut_all=True),
            "cut_for_search": jieba.lcut_for_search(s),
            "cut_for_search_no_hmm": jieba.lcut_for_search(s, HMM=False),
        }
        for key, got in checks.items():
            if got != case[key]:
                mismatches.append((i, s, key, case[key], got))
    assert not mismatches, _fmt(mismatches)


def test_tokenize_parity(segmentation):
    mismatches = []
    for i, case in enumerate(segmentation):
        s = case["s"]
        got_default = [list(t) for t in jieba.tokenize(s)]
        got_search = [list(t) for t in jieba.tokenize(s, mode="search")]
        if got_default != case["tokenize_default"]:
            mismatches.append((i, s, "tokenize_default", case["tokenize_default"], got_default))
        if got_search != case["tokenize_search"]:
            mismatches.append((i, s, "tokenize_search", case["tokenize_search"], got_search))
    assert not mismatches, _fmt(mismatches)


def test_posseg_parity(segmentation):
    mismatches = []
    for i, case in enumerate(segmentation):
        s = case["s"]
        got = [[p.word, p.flag] for p in pseg.cut(s)]
        got_nh = [[p.word, p.flag] for p in pseg.cut(s, HMM=False)]
        if got != case["posseg"]:
            mismatches.append((i, s, "posseg", case["posseg"], got))
        if got_nh != case["posseg_no_hmm"]:
            mismatches.append((i, s, "posseg_no_hmm", case["posseg_no_hmm"], got_nh))
    assert not mismatches, _fmt(mismatches)


def test_jieba_test_corpus_parity(jieba_test_corpus):
    mismatches = []
    for i, case in enumerate(jieba_test_corpus):
        s = case["s"]
        checks = {
            "cut_default": jieba.lcut(s),
            "cut_no_hmm": jieba.lcut(s, HMM=False),
            "cut_all": jieba.lcut(s, cut_all=True),
            "cut_for_search": jieba.lcut_for_search(s),
        }
        for key, got in checks.items():
            if key in case and got != case[key]:
                mismatches.append((i, s, key, case[key], got))
    assert not mismatches, _fmt(mismatches)


def test_extract_tags_parity(analyse_golden):
    import os

    article = open(
        os.path.join(os.path.dirname(__file__), "..", "fixtures", "corpus", "article_zh.txt"),
        encoding="utf-8",
    ).read()

    got = jieba.analyse.extract_tags(article, topK=50, withWeight=True)
    _assert_weighted_equal(got, analyse_golden["extract_tags_top50"])

    got_pos = jieba.analyse.extract_tags(
        article, topK=50, withWeight=True, allowPOS=("ns", "n", "vn", "v")
    )
    _assert_weighted_equal(got_pos, analyse_golden["extract_tags_top50_pos"])


def test_textrank_parity(analyse_golden):
    import os

    article = open(
        os.path.join(os.path.dirname(__file__), "..", "fixtures", "corpus", "article_zh.txt"),
        encoding="utf-8",
    ).read()
    got = jieba.analyse.textrank(article, topK=50, withWeight=True)
    _assert_weighted_equal(got, analyse_golden["textrank_top50"])


def _assert_weighted_equal(got, expected):
    assert len(got) == len(expected), f"length {len(got)} != {len(expected)}"
    for (gw, gwt), (ew, ewt) in zip(got, expected):
        assert gw == ew, f"word order mismatch: {gw!r} != {ew!r}"
        assert gwt == pytest.approx(ewt, rel=1e-9), f"weight for {gw!r}: {gwt} != {ewt}"


def _fmt(mismatches, limit=15):
    lines = [f"{len(mismatches)} mismatches (showing up to {limit}):"]
    for i, s, key, exp, got in mismatches[:limit]:
        lines.append(f"  [{i}] {key} s={s!r}")
        lines.append(f"      expected: {exp}")
        lines.append(f"      got:      {got}")
    return "\n".join(lines)
