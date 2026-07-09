"""T4.1 (mutation) — dictionary-mutation parity against real jieba."""
import os

import jieba_rs as jieba

CORPUS = os.path.join(os.path.dirname(__file__), "..", "fixtures", "corpus")
USERDICT = os.path.join(CORPUS, "userdict.txt")

PROBE_WORDS = [
    "云计算", "韩玉赏鉴", "台中", "石墨烯", "创新办", "凱特琳", "自定义词",
    "中", "将", "北京", "清华大学", "中国", "科学院", "计算所", "李小福",
    "网易", "杭研", "大厦", "叛徒", "不存在的词xyz",
]

SUGGEST_SEGMENTS = [
    "台中", ("中", "将"), "石墨烯", ("今天", "天气"), "清华大学",
    ("北京", "大学"), "中国科学院", "a", ("结婚", "的", "和尚"), "结婚",
]


def seg_repr(seg):
    return seg if isinstance(seg, str) else "|".join(seg)


def test_mutation_sequence(mutation_golden):
    g = mutation_golden
    t = jieba.Tokenizer()
    t.initialize()
    assert t.total == g["initial_total"]

    t.add_word("石墨烯")
    assert t.total == g["after_add_shimoxi_total"]
    assert t.FREQ.get("石墨烯") == g["after_add_shimoxi_freq"]

    t.add_word("凱特琳")
    assert t.total == g["after_add_kailin_total"]
    assert t.FREQ.get("凱特琳") == g["after_add_kailin_freq"]

    t.del_word("自定义词")
    assert t.total == g["after_del_zidingyi_total"]
    assert t.FREQ.get("自定义词") == g["after_del_zidingyi_freq"]

    t.load_userdict(USERDICT)
    assert t.total == g["after_load_userdict_total"]

    probes = {w: t.FREQ.get(w, None) for w in PROBE_WORDS}
    assert probes == g["probe_freq_after_userdict"]

    assert t.lcut("李小福是创新办主任也是云计算方面的专家") == g["mutated_cut_lisiaofu"]
    assert t.lcut("石墨烯是一种材料") == g["mutated_cut_shimoxi"]
    assert t.lcut("韩玉赏鉴很好") == g["mutated_cut_hanyu"]


def test_suggest_freq_samples(mutation_golden):
    t2 = jieba.Tokenizer()
    t2.initialize()
    got = [[seg_repr(seg), t2.suggest_freq(seg, tune=False)] for seg in SUGGEST_SEGMENTS]
    assert got == mutation_golden["suggest_freq_samples"]


def test_suggest_freq_tune_module_level(mutation_golden):
    # Uses a fresh Tokenizer to avoid polluting the shared jieba.dt across tests.
    t = jieba.Tokenizer()
    t.initialize()
    ret = t.suggest_freq(("中", "将"), tune=True)
    expect = mutation_golden["suggest_freq_zhong_jiang"]
    assert ret == expect["return"]
    assert t.lcut("如果放到post中将出错。", HMM=False) == expect["cut_no_hmm"]
