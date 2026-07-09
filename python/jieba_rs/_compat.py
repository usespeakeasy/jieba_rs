"""Compatibility helpers exposed by Python jieba 0.42.1."""

import logging
import os
import sys


log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)


def setLogLevel(log_level):
    default_logger.setLevel(log_level)


def get_module_res(*res):
    return open(os.path.normpath(os.path.join(os.path.dirname(__file__), *res)), "rb")


PY2 = sys.version_info[0] == 2
default_encoding = sys.getfilesystemencoding()
text_type = str
string_types = (str,)
xrange = range
iterkeys = lambda d: iter(d.keys())
itervalues = lambda d: iter(d.values())
iteritems = lambda d: iter(d.items())


def strdecode(sentence):
    if not isinstance(sentence, text_type):
        try:
            sentence = sentence.decode("utf-8")
        except UnicodeDecodeError:
            sentence = sentence.decode("gbk", "ignore")
    return sentence


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)
