"""T4.2 — API surface & behavior parity with jieba's public interface."""
import importlib.util
import inspect

import jieba_rs as jieba
import jieba_rs.analyse
import jieba_rs.posseg as pseg
import pytest


def test_legacy_jieba_import_is_not_provided():
    assert importlib.util.find_spec("jieba") is None


def test_module_attributes():
    assert jieba.__version__ == "0.42.1"
    assert jieba.__license__ == "MIT"
    assert jieba.DEFAULT_DICT is None
    assert jieba.DEFAULT_DICT_NAME == "dict.txt"
    assert isinstance(jieba.user_word_tag_tab, dict)
    assert jieba.pool is None
    # compiled regexes some downstream code imports
    assert jieba.re_han_default.match("北京")
    assert jieba.re_skip_default.match(" ")


@pytest.mark.parametrize(
    "name",
    [
        "cut", "lcut", "cut_for_search", "lcut_for_search", "tokenize",
        "add_word", "del_word", "suggest_freq", "load_userdict", "set_dictionary",
        "initialize", "get_FREQ", "get_DAG", "calc", "enable_parallel",
        "disable_parallel", "setLogLevel",
    ],
)
def test_module_callables_exist(name):
    assert callable(getattr(jieba, name))


@pytest.mark.parametrize(
    "name",
    [
        "cut", "lcut", "cut_for_search", "lcut_for_search", "tokenize",
        "add_word", "del_word", "suggest_freq", "load_userdict", "set_dictionary",
        "initialize", "check_initialized", "get_DAG", "calc",
        "get_dict_file", "__repr__",
    ],
)
def test_tokenizer_methods_exist(name):
    t = jieba.Tokenizer()
    assert hasattr(t, name)


def test_tokenizer_freq_dict():
    # A separate Tokenizer does not initialize the module-level dt.
    t = jieba.Tokenizer()
    t.initialize()
    assert t.FREQ.get("北京") == 34488
    assert jieba.get_FREQ("北京") is None
    jieba.initialize()
    assert jieba.get_FREQ("北京") == 34488
    assert jieba.get_FREQ("不存在xyz", 0) == 0


def test_cut_signature_kwargs():
    sig = inspect.signature(jieba.Tokenizer.cut)
    params = sig.parameters
    assert "cut_all" in params and params["cut_all"].default is False
    assert "HMM" in params and params["HMM"].default is True
    assert "use_paddle" not in params


def test_removed_optional_integrations_are_absent():
    assert not hasattr(jieba, "enable_paddle")
    assert importlib.util.find_spec("jieba_rs.lac_small") is None
    assert importlib.util.find_spec("jieba_rs.analyse.analyzer") is None
    assert not hasattr(jieba.analyse, "ChineseAnalyzer")


def test_posseg_surface():
    assert callable(pseg.cut)
    assert callable(pseg.lcut)
    p = pseg.pair("北京", "ns")
    assert p.word == "北京" and p.flag == "ns"
    assert str(p) == "北京/ns"
    assert list(p) == ["北京", "ns"]
    assert p == pseg.pair("北京", "ns")
    assert hash(p) == hash(pseg.pair("北京", "x"))  # hash by word only
    assert isinstance(pseg.dt, pseg.POSTokenizer)


def test_analyse_surface():
    for name in ["extract_tags", "textrank", "set_idf_path", "set_stop_words"]:
        assert callable(getattr(jieba.analyse, name))
    assert isinstance(jieba.analyse.TFIDF(), jieba.analyse.TFIDF)
    assert isinstance(jieba.analyse.TextRank(), jieba.analyse.TextRank)
    assert jieba.analyse.default_tfidf is not None
    assert jieba.analyse.default_textrank is not None
