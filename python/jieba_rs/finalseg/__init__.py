"""Python-compatible finalseg module backed by jieba-rs's bundled HMM data."""

from __future__ import absolute_import, unicode_literals

import json
import os
import re

from .._compat import *


MIN_FLOAT = -3.14e100
PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
PrevStatus = {"B": "ES", "M": "MB", "S": "SE", "E": "BM"}
Force_Split_Words = set()


def _load_json_model():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hmm_data.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["finalseg"]


_MODEL = None
start_P = None
trans_P = None
emit_P = None


def _ensure_model():
    global _MODEL, start_P, trans_P, emit_P
    if _MODEL is None:
        _MODEL = _load_json_model()
        start_P = _MODEL["start"]
        trans_P = _MODEL["trans"]
        emit_P = _MODEL["emit"]
    return start_P, trans_P, emit_P


def load_model():
    return _ensure_model()


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in xrange(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            prob, state = max(
                (V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0)
                for y0 in PrevStatus[y]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    prob, state = max((V[len(obs) - 1][y], y) for y in "ES")
    return prob, path[state]


def __cut(sentence):
    start, trans, emit = _ensure_model()
    _, pos_list = viterbi(sentence, "BMES", start, trans, emit)
    begin, nexti = 0, 0
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == "B":
            begin = i
        elif pos == "E":
            yield sentence[begin : i + 1]
            nexti = i + 1
        elif pos == "S":
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]


re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile(r"([a-zA-Z0-9]+(?:\.\d+)?%?)")


def add_force_split(word):
    Force_Split_Words.add(word)
    try:
        import jieba_rs as jieba

        sync = getattr(jieba, "_sync_force_split", None)
        if sync is not None:
            sync(word)
    except ImportError:
        pass


def cut(sentence):
    sentence = strdecode(sentence)
    for blk in re_han.split(sentence):
        if re_han.match(blk):
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    yield from word
        else:
            for x in re_skip.split(blk):
                if x:
                    yield x


from . import prob_emit, prob_start, prob_trans  # noqa: E402,F401
