"""jieba_rs — Rust-backed Chinese segmentation for the jieba-rs distribution.

Segmentation runs in the Rust core (`jieba_rs._rustjieba`); this module exposes
a focused tokenizer API.
"""
from __future__ import absolute_import, unicode_literals

import logging
import marshal
import os
import re
import sys
import tempfile
import threading
import time
import weakref
from hashlib import md5
from math import log

from ._rustjieba import RustTokenizer
from . import finalseg
from ._compat import *

__version__ = "0.42.1"
__license__ = "MIT"

DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DEFAULT_DICT_PATH = os.path.join(_DATA_DIR, "dict.txt")
_HMM_PATH = os.path.join(_DATA_DIR, "hmm_data.json")
_IDF_PATH = os.path.join(_DATA_DIR, "idf.txt")

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.WARNING)
default_logger.addHandler(log_console)

pool = None
DICT_WRITING = {}
try:
    _ALL_TOKENIZERS
except NameError:
    _ALL_TOKENIZERS = weakref.WeakSet()

if os.name == "nt":
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

re_userdict = re.compile("^(.+?)( [0-9]+)?( [a-z]+)?$", re.U)
re_eng = re.compile("[a-zA-Z0-9]", re.U)
re_han_default = re.compile("([一-鿕a-zA-Z0-9+#&\\._%\\-]+)", re.U)
re_skip_default = re.compile("(\\r\\n|\\s)", re.U)

def setLogLevel(log_level):
    default_logger.setLevel(log_level)


def strdecode(sentence):
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


class _FreqDict(dict):
    """A real dict that mirrors direct mutations into the Rust dictionary."""

    def __init__(self, tokenizer):
        super().__init__()
        self._tk = tokenizer

    def _replace(self, values):
        dict.clear(self)
        dict.update(self, values)

    def _set_local(self, key, value):
        dict.__setitem__(self, key, value)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if self._tk.initialized and isinstance(key, str):
            try:
                rust_value = int(value)
            except (TypeError, ValueError, OverflowError):
                return
            self._tk._rust.set_freq(key, rust_value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        if self._tk.initialized and isinstance(key, str):
            self._tk._rust.delete_freq(key)

    def clear(self):
        dict.clear(self)
        if self._tk.initialized:
            self._tk._rust.clear_freq()

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError("pop expected at most 2 arguments, got %d" % (len(default) + 1))
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key, value = next(reversed(self.items()))
        del self[key]
        return key, value

    def copy(self):
        return dict(self)

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.get(self, key, default)
        if self._tk.initialized and isinstance(key, str):
            value = self._tk._rust.get_freq(key)
            return default if value is None else value
        return default

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        value = self.get(key, None)
        if value is None:
            raise KeyError(key)
        return value

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return dict.fromkeys(iterable, value)

    def __reduce__(self):
        return dict, (dict(self),)

    def __ior__(self, other):
        self.update(other)
        return self


class Tokenizer(object):
    def __setattr__(self, name, value):
        if (
            name == "FREQ"
            and hasattr(self, "_rust")
            and (not isinstance(value, _FreqDict) or value._tk is not self)
        ):
            freq = _FreqDict(self)
            freq._replace(value)
            object.__setattr__(self, name, freq)
            if getattr(self, "initialized", False):
                entries = []
                for key, item in freq.items():
                    if isinstance(key, str):
                        try:
                            entries.append((key, int(item)))
                        except (TypeError, ValueError, OverflowError):
                            pass
                self._rust.set_freq_entries(entries)
            return
        object.__setattr__(self, name, value)
        if name == "total" and getattr(self, "initialized", False):
            try:
                rust_value = int(value)
            except (TypeError, ValueError, OverflowError):
                return
            self._rust.set_total(rust_value)

    def __init__(self, dictionary=DEFAULT_DICT):
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = DEFAULT_DICT
            dict_path = _DEFAULT_DICT_PATH
        else:
            self.dictionary = _get_abs_path(dictionary)
            dict_path = self.dictionary
        self._rust = RustTokenizer(dict_path, _HMM_PATH, _IDF_PATH)
        self.initialized = False
        self.FREQ = _FreqDict(self)
        self.total = 0
        self.user_word_tag_tab = {}
        self.tmp_dir = None
        self.cache_file = None
        _ALL_TOKENIZERS.add(self)
        self._force_snapshot = frozenset(finalseg.Force_Split_Words)
        self._rust.set_force_split_words(list(self._force_snapshot))

    def __repr__(self):
        return "<Tokenizer dictionary=%r>" % self.dictionary

    @staticmethod
    def gen_pfdict(f):
        lfreq = {}
        ltotal = 0
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                word, freq = line.split(" ")[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq
                for ch in range(len(word)):
                    wfrag = word[: ch + 1]
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                raise ValueError(
                    "invalid dictionary entry in %s at Line %s: %s" % (f_name, lineno, line)
                )
        f.close()
        return lfreq, ltotal

    # ---- lifecycle ----

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = _get_abs_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            self.dictionary = abs_path
            self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:
            if self.initialized:
                return

            default_logger.debug(
                "Building prefix dict from %s ..." % (abs_path or "the default dictionary")
            )
            started = time.time()
            if abs_path != DEFAULT_DICT:
                self._rust.set_dictionary(abs_path)
            self._rust.initialize()
            loaded_freq = _FreqDict(self)
            object.__setattr__(self, "FREQ", loaded_freq)
            self.total = self._rust.total
            self.initialized = True
            default_logger.debug("Loading model cost %.3f seconds." % (time.time() - started))
            default_logger.debug("Prefix dict has been built successfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def set_dictionary(self, dictionary_path):
        with self.lock:
            abs_path = _get_abs_path(dictionary_path)
            if not os.path.isfile(abs_path):
                raise Exception("jieba-rs: file does not exist: " + abs_path)
            self.dictionary = abs_path
            self._rust.set_dictionary(abs_path)
            self.initialized = False

    # ---- segmentation (generators, like jieba) ----

    def _sync_finalseg_state(self):
        current = frozenset(finalseg.Force_Split_Words)
        if current != self._force_snapshot:
            self._rust.set_force_split_words(list(current))
            self._force_snapshot = current

    def __cut_all(self, sentence):
        self.check_initialized()
        self._sync_finalseg_state()
        yield from self._rust.cut(sentence, True, True)

    def __cut_DAG_NO_HMM(self, sentence):
        self.check_initialized()
        self._sync_finalseg_state()
        yield from self._rust.cut(sentence, False, False)

    def __cut_DAG(self, sentence):
        self.check_initialized()
        self._sync_finalseg_state()
        yield from self._rust.cut(sentence, False, True)

    def cut(self, sentence, cut_all=False, HMM=True):
        sentence = strdecode(sentence)
        self.check_initialized()
        self._sync_finalseg_state()
        yield from self._rust.cut(sentence, cut_all, HMM)

    def cut_for_search(self, sentence, HMM=True):
        sentence = strdecode(sentence)
        self.check_initialized()
        self._sync_finalseg_state()
        yield from self._rust.cut_for_search(sentence, HMM)

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    _lcut = lcut
    _lcut_for_search = lcut_for_search

    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba-rs: the input parameter should be unicode.")
        self.check_initialized()
        self._sync_finalseg_state()
        search = mode != "default"
        result = self._rust.tokenize(unicode_sentence, search, HMM)
        for token in result:
            yield tuple(token)

    # ---- DAG internals (parity surface) ----

    def get_DAG(self, sentence):
        self.check_initialized()
        dag_list = self._rust.get_dag(sentence)
        return {k: list(ends) for k, ends in enumerate(dag_list)}

    def calc(self, sentence, DAG, route):
        self.check_initialized()
        n = len(DAG)
        dag_list = [list(DAG[k]) for k in range(n)]
        computed = self._rust.calc(sentence, dag_list)
        route[n] = (0, 0)
        for idx in range(n - 1, -1, -1):
            logp, x = computed[idx]
            route[idx] = (logp, x)

    # ---- dictionary mutation ----

    def add_word(self, word, freq=None, tag=None):
        self.check_initialized()
        word = strdecode(word)
        rfreq = int(freq) if freq is not None else self.suggest_freq(word, False)
        self._rust.add_word(word, rfreq, tag)
        self.FREQ._set_local(word, rfreq)
        self.total += rfreq
        if tag:
            self.user_word_tag_tab[word] = tag
        for ch in range(len(word)):
            wfrag = word[: ch + 1]
            if wfrag not in self.FREQ:
                self.FREQ._set_local(wfrag, 0)
        if rfreq == 0:
            finalseg.add_force_split(word)

    def del_word(self, word):
        self.add_word(word, 0)

    def suggest_freq(self, segment, tune=False):
        self.check_initialized()
        if isinstance(segment, string_types):
            word = strdecode(segment)
            freq = self._rust.suggest_freq_word(word, tune)
        else:
            parts = tuple(strdecode(s) for s in segment)
            word = "".join(parts)
            freq = self._rust.suggest_freq_parts(list(parts), tune)
        if tune:
            self.FREQ._set_local(word, freq)
            self.total = self._rust.total
            if freq == 0:
                finalseg.add_force_split(word)
        return freq

    def load_userdict(self, f):
        self.check_initialized()
        if isinstance(f, string_types):
            f_name = f
            f = open(f, "rb")
        else:
            f_name = resolve_filename(f)
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not isinstance(line, text_type):
                try:
                    line = line.decode("utf-8").lstrip("\ufeff")
                except UnicodeDecodeError:
                    raise ValueError("dictionary file %s must be utf-8" % f_name)
            if not line:
                continue
            word, freq, tag = re_userdict.match(line).groups()
            self.add_word(
                word,
                freq.strip() if freq is not None else None,
                tag.strip() if tag is not None else None,
            )

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return open(_DEFAULT_DICT_PATH, "rb")
        return open(self.dictionary, "rb")


def _sync_force_split(word):
    for tokenizer in list(_ALL_TOKENIZERS):
        tokenizer._rust.add_force_split(word)
        tokenizer._force_snapshot = frozenset(finalseg.Force_Split_Words)


# default Tokenizer instance
dt = Tokenizer()

# module-level functions delegating to dt
get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
add_word = dt.add_word
calc = dt.calc
cut = dt.cut
lcut = dt.lcut
cut_for_search = dt.cut_for_search
lcut_for_search = dt.lcut_for_search
del_word = dt.del_word
get_DAG = dt.get_DAG
get_dict_file = dt.get_dict_file
initialize = dt.initialize
load_userdict = dt.load_userdict
set_dictionary = dt.set_dictionary
suggest_freq = dt.suggest_freq
tokenize = dt.tokenize
user_word_tag_tab = dt.user_word_tag_tab


def _lcut_all(s):
    return dt._lcut_all(s)


def _lcut(s):
    return dt._lcut(s)


def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)


def _lcut_for_search(s):
    return dt._lcut_for_search(s)


def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)


def enable_parallel(processnum=None):
    global pool, cut, cut_for_search
    from multiprocessing import Pool, cpu_count

    if os.name == "nt":
        raise NotImplementedError("jieba-rs: parallel mode only supports posix system")
    dt.check_initialized()
    if processnum is None:
        processnum = cpu_count()
    pool = Pool(processnum)
    cut = _pcut
    cut_for_search = _pcut_for_search


def disable_parallel():
    global pool, cut, cut_for_search
    if pool:
        pool.close()
        pool = None
    cut = dt.cut
    cut_for_search = dt.cut_for_search


def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for row in result:
        yield from row


def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    fn = _lcut_for_search if HMM else _lcut_for_search_no_hmm
    for row in pool.map(fn, parts):
        yield from row
