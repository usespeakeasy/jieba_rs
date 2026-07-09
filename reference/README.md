# Reference implementation (not shipped)

Consult jieba 0.42.1 source here for exact parity. Re-clone with:

    git clone --depth 1 --branch v0.42.1 https://github.com/fxsjy/jieba reference/jieba

This directory is gitignored (except this README). It is a build-time/consultation
resource only. Data files (dict.txt, idf.txt, HMM tables) were copied into
python/jieba_rs/data/ and remain under jieba's MIT license.
