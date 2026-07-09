import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "fixtures", "golden")
CORPUS = os.path.join(ROOT, "fixtures", "corpus")


def _load(name):
    with open(os.path.join(GOLDEN, name), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def segmentation():
    return _load("segmentation.json")


@pytest.fixture(scope="session")
def analyse_golden():
    return _load("analyse.json")


@pytest.fixture(scope="session")
def mutation_golden():
    return _load("mutation.json")


@pytest.fixture(scope="session")
def jieba_test_corpus():
    return _load("jieba_test_corpus.json")
