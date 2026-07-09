"""T4.2 — behavioral parity: generators, errors, instance isolation."""
import os

import jieba_rs as jieba
import jieba_rs.posseg as pseg
import pytest

CORPUS = os.path.join(os.path.dirname(__file__), "..", "fixtures", "corpus")
USERDICT = os.path.join(CORPUS, "userdict.txt")


def test_cut_returns_iterator_not_list():
    g = jieba.cut("我来到北京清华大学")
    assert not isinstance(g, list)
    assert hasattr(g, "__next__")
    assert "".join(g)  # consumable


def test_cut_for_search_and_tokenize_are_iterators():
    assert hasattr(jieba.cut_for_search("北京大学"), "__next__")
    assert hasattr(jieba.tokenize("北京大学"), "__next__")
    assert hasattr(pseg.cut("北京大学"), "__next__")


def test_tokenize_bytes_raises():
    with pytest.raises(ValueError, match="unicode"):
        list(jieba.tokenize(b"abc"))


def test_independent_instances_do_not_share_mutations():
    a = jieba.Tokenizer()
    b = jieba.Tokenizer()
    a.initialize()
    b.initialize()
    a.add_word("石墨烯特殊词", freq=99999)
    assert a.FREQ.get("石墨烯特殊词") == 99999
    assert b.FREQ.get("石墨烯特殊词") is None


def test_load_userdict_accepts_path_and_filelike():
    t1 = jieba.Tokenizer()
    t1.initialize()
    t1.load_userdict(USERDICT)
    assert t1.FREQ.get("云计算") == 5

    t2 = jieba.Tokenizer()
    t2.initialize()
    with open(USERDICT, "r", encoding="utf-8") as fh:
        t2.load_userdict(fh)
    assert t2.FREQ.get("云计算") == 5


def test_set_dictionary_before_first_cut(tmp_path):
    # A tiny custom dictionary; set before any cut triggers lazy load of it.
    d = tmp_path / "mini.txt"
    d.write_text("北京 100 ns\n清华 50 n\n", encoding="utf-8")
    t = jieba.Tokenizer()
    t.set_dictionary(str(d))
    assert t.lcut("北京清华") == ["北京", "清华"]
    assert t.FREQ.get("北京") == 100


def test_pair_equality_and_str():
    p = pseg.pair("中国", "ns")
    assert str(p) == "中国/ns"
    assert repr(p) == "pair('中国', 'ns')"
    w, f = p
    assert (w, f) == ("中国", "ns")
