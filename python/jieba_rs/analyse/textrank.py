from __future__ import absolute_import, unicode_literals

import sys
from collections import defaultdict
from operator import itemgetter

import jieba_rs as jieba
import jieba_rs.posseg

from .tfidf import KeywordExtractor


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)
        sorted_keys = sorted(self.graph.keys())
        for _ in range(10):
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s
        min_rank, max_rank = sys.float_info[0], sys.float_info[3]
        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w
        for n, w in ws.items():
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)
        return ws


class TextRank(KeywordExtractor):
    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(("ns", "n", "vn", "v"))
        self.span = 5

    def pairfilter(self, wp):
        return (
            wp.flag in self.pos_filt
            and len(wp.word.strip()) >= 2
            and wp.word.lower() not in self.stop_words
        )

    def textrank(
        self,
        sentence,
        topK=20,
        withWeight=False,
        allowPOS=("ns", "n", "vn", "v"),
        withFlag=False,
    ):
        self.pos_filt = frozenset(allowPOS)
        graph = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))
        for i, wp in enumerate(words):
            if self.pairfilter(wp):
                for j in range(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1
        for terms, weight in cm.items():
            graph.addEdge(terms[0], terms[1], weight)
        nodes_rank = graph.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)
        return tags[:topK] if topK else tags

    extract_tags = textrank
