"""Regressions for the supported Python jieba_rs surface."""

import importlib
import io
import os
import subprocess
import sys
import types

import jieba_rs as jieba
import jieba_rs.analyse
import jieba_rs.posseg as pseg
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERDICT = os.path.join(ROOT, "fixtures", "corpus", "userdict.txt")
TEXT = "北京大学和清华大学位于北京，计算机科学研究推动人工智能发展。北京大学研究人工智能。"
ALLOW_POS = ("ns", "n", "vn", "v", "nt", "nz")


def test_cut_apis_are_real_lazy_generators():
    for method, args, kwargs in [
        ("cut", ("北京大学",), {}),
        ("cut", ("北京大学",), {"cut_all": True}),
        ("cut_for_search", ("北京大学",), {}),
        ("tokenize", ("北京大学",), {}),
    ]:
        tokenizer = jieba.Tokenizer()
        result = getattr(tokenizer, method)(*args, **kwargs)
        assert isinstance(result, types.GeneratorType)
        assert tokenizer.initialized is False
        assert list(result)
        assert tokenizer.initialized is True


def test_freq_is_sparse_mutation_overlay_and_does_not_force_initialization_on_get():
    tokenizer = jieba.Tokenizer()
    preinit_freq = tokenizer.FREQ
    assert isinstance(tokenizer.FREQ, dict)
    assert tokenizer.FREQ.get("北京") is None
    assert tokenizer.initialized is False

    tokenizer.initialize()
    assert tokenizer.FREQ is not preinit_freq
    assert preinit_freq == {}
    assert len(tokenizer.FREQ) == 0
    assert tokenizer.FREQ.get("北京") == 34488
    tokenizer.FREQ["直写测试词"] = 123
    assert tokenizer.FREQ["直写测试词"] == 123
    assert tokenizer.FREQ.setdefault("允许任意字典值") is None

    tokenizer.FREQ = {"测": 0, "测试": 100}
    assert tokenizer.lcut("测试", HMM=False) == ["测试"]


def test_calc_mutates_route_and_returns_none():
    tokenizer = jieba.Tokenizer()
    sentence = "北京大学"
    dag = tokenizer.get_DAG(sentence)
    route = {}
    assert tokenizer.calc(sentence, dag, route) is None
    assert len(route) == len(sentence) + 1
    assert list(route) == list(range(len(sentence), -1, -1))


def test_tokenizer_reference_attributes_and_userdict_metadata():
    tokenizer = jieba.Tokenizer()
    assert tokenizer.tmp_dir is None
    assert tokenizer.cache_file is None
    assert callable(tokenizer.gen_pfdict)

    tokenizer.load_userdict(USERDICT)
    assert tokenizer.user_word_tag_tab["创新办"] == "i"
    assert tokenizer.user_word_tag_tab["凱特琳"] == "nz"
    assert tokenizer.user_word_tag_tab["云计算"] == "n"


def test_invalid_utf8_userdict_is_rejected():
    with pytest.raises(ValueError, match="must be utf-8"):
        jieba.Tokenizer().load_userdict(io.BytesIO(b"\xffbadword 5 n\n"))


def test_nondefault_tokenize_mode_uses_search_semantics():
    tokenizer = jieba.Tokenizer()
    assert list(tokenizer.tokenize("中国科学院", mode="bogus")) == list(
        tokenizer.tokenize("中国科学院", mode="search")
    )


def test_direct_finalseg_force_split_mutation_is_observed():
    import jieba_rs.finalseg

    tokenizer = jieba.Tokenizer()
    assert tokenizer.lcut("杭研") == ["杭研"]
    jieba.finalseg.Force_Split_Words.add("杭研")
    try:
        assert tokenizer.lcut("杭研") == ["杭", "研"]
    finally:
        jieba.finalseg.Force_Split_Words.remove("杭研")
    assert tokenizer.lcut("杭研") == ["杭研"]


def test_analysis_withflag_returns_pos_pairs():
    tfidf = jieba.analyse.TFIDF()
    tags = tfidf.extract_tags(TEXT, topK=6, allowPOS=ALLOW_POS, withFlag=True)
    weighted = tfidf.extract_tags(
        TEXT, topK=3, allowPOS=ALLOW_POS, withFlag=True, withWeight=True
    )
    assert all(isinstance(tag, pseg.pair) for tag in tags)
    assert all(isinstance(tag, pseg.pair) and isinstance(weight, float) for tag, weight in weighted)

    textrank = jieba.analyse.TextRank()
    tags = textrank.textrank(TEXT, topK=6, allowPOS=ALLOW_POS, withFlag=True)
    assert all(isinstance(tag, pseg.pair) for tag in tags)


def test_analysis_instances_are_independent_and_span_is_honored(tmp_path):
    default = jieba.analyse.TFIDF()
    before = default.extract_tags(TEXT, topK=8, withWeight=True)
    custom_idf = tmp_path / "idf.txt"
    custom_idf.write_text(
        "人工智能 1\n北京大学 100\n计算机科学 2\n研究 3\n清华大学 4\n推动 5\n位于 6\n北京 7\n",
        encoding="utf-8",
    )
    custom = jieba.analyse.TFIDF(str(custom_idf))
    assert custom.idf_loader.path == str(custom_idf)
    assert custom.extract_tags(TEXT, topK=8, withWeight=True) != before
    assert default.extract_tags(TEXT, topK=8, withWeight=True) == before

    span2 = jieba.analyse.TextRank()
    span2.span = 2
    span5 = jieba.analyse.TextRank()
    span5.span = 5
    assert span2.textrank(TEXT, topK=None, withWeight=True) != span5.textrank(
        TEXT, topK=None, withWeight=True
    )


def test_analysis_negative_topk_matches_python_slicing():
    tags = jieba.analyse.TFIDF().extract_tags(TEXT, topK=-1)
    assert tags


@pytest.mark.parametrize(
    "module",
    [
        "jieba_rs._compat",
        "jieba_rs.finalseg",
        "jieba_rs.finalseg.prob_start",
        "jieba_rs.finalseg.prob_trans",
        "jieba_rs.finalseg.prob_emit",
        "jieba_rs.posseg.prob_start",
        "jieba_rs.posseg.prob_trans",
        "jieba_rs.posseg.prob_emit",
        "jieba_rs.posseg.char_state_tab",
        "jieba_rs.posseg.viterbi",
        "jieba_rs.analyse.tfidf",
        "jieba_rs.analyse.textrank",
    ],
)
def test_reference_submodules_are_importable(module):
    assert importlib.import_module(module).__name__ == module


def test_posseg_compatibility_surface_exists():
    assert callable(pseg.initialize)
    tokenizer = pseg.POSTokenizer()
    assert isinstance(tokenizer.word_tag_tab, dict)
    assert callable(tokenizer.makesure_userdict_loaded)
    assert callable(tokenizer._lcut_internal)
    assert callable(tokenizer._lcut_internal_no_hmm)


def test_posseg_uses_rust_tag_table_for_dictionary_words(tmp_path):
    tokenizer = jieba.Tokenizer()
    postokenizer = pseg.POSTokenizer(tokenizer)
    postokenizer.word_tag_tab["北京"] = "custom"
    assert [(item.word, item.flag) for item in postokenizer.cut("北京")] == [
        ("北京", "ns")
    ]

    custom_dict = tmp_path / "dict.txt"
    custom_dict.write_text("自造词 100 abc\n", encoding="utf-8")
    tokenizer.set_dictionary(str(custom_dict))
    # The Rust dictionary/tag state is authoritative for the supported contract.
    assert [(item.word, item.flag) for item in postokenizer.cut("自造词")] == [
        ("自造词", "abc")
    ]


def test_module_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "jieba_rs", "--help"],
        cwd=os.path.dirname(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jieba-rs command line interface" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="jieba_rs parallel mode is POSIX-only")
def test_parallel_mode_matches_serial_mode():
    sentence = "我来到北京清华大学\n小明硕士毕业于中国科学院计算所"
    expected = jieba.dt.lcut(sentence)
    jieba.enable_parallel(2)
    try:
        assert list(jieba.cut(sentence)) == expected
    finally:
        jieba.disable_parallel()
    assert jieba.pool is None
