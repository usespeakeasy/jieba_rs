"""Core regression scenarios adapted from jieba v0.42.1's public tests."""

from concurrent.futures import ThreadPoolExecutor

import jieba_rs


UPSTREAM_SENTENCES = (
    "我爱北京天安门",
    "中国科学院计算技术研究所",
    "结婚的和尚未结婚的",
    "南京市长江大桥",
    "长春市长春药店",
    "AT&T是一件不错的公司，给你发offer了吗？",
)


def test_shared_tokenizer_is_safe_under_concurrent_cuts():
    tokenizer = jieba_rs.Tokenizer()
    expected = {sentence: tokenizer.lcut(sentence) for sentence in UPSTREAM_SENTENCES}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(tokenizer.lcut, sentence)
            for _ in range(8)
            for sentence in UPSTREAM_SENTENCES
        ]

    for index, future in enumerate(futures):
        sentence = UPSTREAM_SENTENCES[index % len(UPSTREAM_SENTENCES)]
        assert future.result() == expected[sentence]


def test_cut_accepts_text_file_content(tmp_path):
    source = tmp_path / "test.txt"
    source.write_text("我爱北京天安门\n中国科学院计算技术研究所\n", encoding="utf-8")

    lines = source.read_text(encoding="utf-8").splitlines()
    assert ["".join(jieba_rs.cut(line)) for line in lines] == lines
