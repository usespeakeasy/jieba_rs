"""jieba_rs.posseg — POS tagging for jieba-rs."""
from __future__ import absolute_import, unicode_literals

import os
import re
import sys

import jieba_rs as jieba
from .._compat import *


PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

char_state_tab_P = None
start_P = None
trans_P = None
emit_P = None


def load_model():
    global char_state_tab_P, start_P, trans_P, emit_P
    if char_state_tab_P is None:
        from . import char_state_tab, prob_emit, prob_start, prob_trans

        char_state_tab_P = char_state_tab.P
        start_P = prob_start.P
        trans_P = prob_trans.P
        emit_P = prob_emit.P
    return char_state_tab_P, start_P, trans_P, emit_P


re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile(r"([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile(r"(\r\n|\s)")
re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile(r"[\.0-9]+")
re_eng1 = re.compile("^[a-zA-Z0-9]$", re.U)


class pair(object):
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return "%s/%s" % (self.word, self.flag)

    def __repr__(self):
        return "pair(%r, %r)" % (self.word, self.flag)

    def __str__(self):
        return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other, pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)

    def encode(self, arg):
        return self.__unicode__().encode(arg)


class POSTokenizer(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba.Tokenizer()
        self.word_tag_tab = {}

    def __repr__(self):
        return "<POSTokenizer tokenizer=%r>" % self.tokenizer

    def __getattr__(self, name):
        if name in ("cut_for_search", "lcut_for_search", "tokenize"):
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def initialize(self, dictionary=None):
        self.tokenizer.initialize(dictionary)
        self.word_tag_tab = {}

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError(
                    "invalid POS dictionary entry in %s at Line %s: %s"
                    % (f_name, lineno, line)
                )
        f.close()

    def makesure_userdict_loaded(self):
        return None

    def cut(self, sentence, HMM=True):
        yield from self.__cut_internal(sentence, HMM=HMM)

    def __cut(self, sentence):
        from .viterbi import viterbi

        char_state_tab, start, trans, emit = load_model()
        _, pos_list = viterbi(sentence, char_state_tab, start, trans, emit)
        begin, nexti = 0, 0
        for index, char in enumerate(sentence):
            pos = pos_list[index][0]
            if pos == "B":
                begin = index
            elif pos == "E":
                yield pair(sentence[begin : index + 1], pos_list[index][1])
                nexti = index + 1
            elif pos == "S":
                yield pair(char, pos_list[index][1])
                nexti = index + 1
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def __cut_detail(self, sentence):
        for block in re_han_detail.split(sentence):
            if re_han_detail.match(block):
                yield from self.__cut(block)
            else:
                for item in re_skip_detail.split(block):
                    if item:
                        if re_num.match(item):
                            yield pair(item, "m")
                        elif re_eng.match(item):
                            yield pair(item, "eng")
                        else:
                            yield pair(item, "x")

    def __cut_DAG_NO_HMM(self, sentence):
        self.tokenizer.check_initialized()
        for word, flag, from_dictionary in self.tokenizer._rust.pos_cut_detailed(
            strdecode(sentence), False
        ):
            yield pair(word, flag)

    def __cut_DAG(self, sentence):
        self.tokenizer.check_initialized()
        for word, flag, from_dictionary in self.tokenizer._rust.pos_cut_detailed(
            strdecode(sentence), True
        ):
            yield pair(word, flag)

    def __cut_internal(self, sentence, HMM=True):
        self.makesure_userdict_loaded()
        self.tokenizer.check_initialized()
        sentence = strdecode(sentence)
        pairs = self.tokenizer._rust.pos_cut_detailed(sentence, HMM)
        for word, flag, from_dictionary in pairs:
            yield pair(word, flag)

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def _lcut_internal(self, sentence):
        return self.lcut(sentence)

    def _lcut_internal_no_hmm(self, sentence):
        return self.lcut(sentence, HMM=False)


# default POSTokenizer bound to jieba_rs.dt
dt = POSTokenizer(jieba.dt)
initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    if jieba.pool is None:
        yield from dt.cut(sentence, HMM=HMM)
    else:
        parts = strdecode(sentence).splitlines(True)
        fn = _lcut_internal if HMM else _lcut_internal_no_hmm
        for row in jieba.pool.map(fn, parts):
            yield from row


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
