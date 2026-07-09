# jieba 0.42.1 Algorithm & Parity Reference

This document captures the exact behavior the Rust core must replicate. The pseudocode below mirrors jieba 0.42.1's source closely, but the vendored copy at `reference/jieba/` is the final authority — **when this doc and the source disagree, the source wins; fix this doc.**

All string indices below are **Unicode scalar values (chars), not bytes**. In Rust, work on `Vec<char>` (or char-index maps); Python `len(str)` counts chars.

---

## §1 Dictionary format & prefix dict

`dict.txt`: one entry per line, `word freq tag`, separated by single spaces. Parsing (from `gen_pfdict`):

```
for each line: strip; split on ' '; word = field[0]; freq = int(field[1])   # tag ignored here
FREQ[word] = freq
total += freq
for each proper prefix p of word (by chars, cumulative):
    if p not in FREQ: FREQ[p] = 0        # marker entry — enables DAG scan termination
```

- With the bundled dict: `total == 60101967`, 349,046 word lines.
- A `FREQ` value of `0` means "is a prefix or deleted word, not a real word". Truthiness checks (`if FREQ[frag]:`) distinguish these.
- POS tags from `dict.txt` are loaded separately by posseg into `word_tag_tab` (`word, _, tag = line.split(" ")`).

## §2 DAG + max-probability route

`get_DAG(sentence)` — for each start index `k`, list of end indices `i` such that `sentence[k..=i]` is a real word (freq > 0). Scan stops at the first fragment not present in `FREQ` at all (prefix markers keep the scan alive). If no word found, `DAG[k] = [k]`.

```
for k in 0..N:
    tmplist = []; i = k; frag = sentence[k..=i]
    while i < N and frag in FREQ:
        if FREQ[frag] > 0: tmplist.push(i)
        i += 1; frag = sentence[k..=i]
    if tmplist empty: tmplist.push(k)
    DAG[k] = tmplist
```

`calc(sentence, DAG)` — right-to-left DP over log probabilities:

```
route[N] = (0.0, 0)
logtotal = ln(total)                               # f64
for idx in (N-1)..=0 descending:
    route[idx] = max over x in DAG[idx] of:
        ( ln( FREQ.get(sentence[idx..=x]) or 1 ) - logtotal + route[x+1].0 , x )
```

**Critical parity details:**
- `FREQ.get(w) or 1`: missing **or zero** freq → 1 (i.e. `ln(1) = 0`).
- Python's `max` over `(value, x)` tuples breaks value-ties by **larger `x`** (longer word wins). In Rust: keep candidate if `v > best.0 || (v == best.0 && x > best.1)`.
- Use `f64` and `f64::ln`. CPython uses the platform `libm`; bit-level differences are theoretically possible — the fixture suite (T4.1) is the arbiter.

## §3 The four cut paths

Regexes (Python syntax; `re_han_*` are used with `re.split` capturing group semantics — the separators between matches are also returned):

```
re_han_default  = ([一-鿕a-zA-Z0-9+#&\._%\-]+)
re_skip_default = (\r\n|\s)
re_han_cut_all  = ([一-鿕]+)
re_skip_cut_all = [^a-zA-Z0-9+#\n]        # NOTE: no capturing group
re_eng          = [a-zA-Z0-9]             # used with .match → prefix match
```

**Rust note on `re.split` with a capturing group:** `"abc中文def".split(re_han)` yields `["", "abc中文def", ""]`-style alternating non-match/match segments. Implement a helper `split_keep(re, s) -> Vec<(&str, bool /*is_match*/)>`. When the regex has **no** capturing group (`re_skip_cut_all`), the separators are dropped by `re.split` — replicate that too.

**Top-level `cut(sentence, cut_all, HMM)`:**

```
choose (re_han, re_skip, block_fn):
    cut_all      → (re_han_cut_all, re_skip_cut_all, cut_all_block)
    HMM          → (re_han_default, re_skip_default, cut_dag_hmm)
    else         → (re_han_default, re_skip_default, cut_dag_no_hmm)
for blk in re_han.split(sentence):
    if blk empty: continue
    if re_han.match(blk):              # match = anchored at start
        yield from block_fn(blk)
    else:
        for x in re_skip.split(blk):
            if re_skip.match(x):       yield x
            elif not cut_all:          yield each char of x     # one token per char
            else:                      yield x
```

**`cut_dag_no_hmm(block)`** — DAG route + buffering of single-char alphanumerics:

```
run get_DAG + calc
x = 0; buf = ""
while x < N:
    y = route[x].1 + 1
    word = sentence[x..y]
    if word is 1 char and re_eng matches it:  buf += word
    else:
        if buf: yield buf; buf = ""
        yield word
    x = y
if buf: yield buf
```

**`cut_dag_hmm(block)`** — DAG route; consecutive single chars are buffered and re-segmented by HMM when the buffer is not itself a dictionary word:

```
run get_DAG + calc
x = 0; buf = ""
while x < N:
    y = route[x].1 + 1
    word = sentence[x..y]
    if y - x == 1: buf += word
    else:
        if buf: flush(buf); buf = ""
        yield word
    x = y
if buf: flush(buf)

flush(buf):
    if len(buf) == 1:                 yield buf
    elif FREQ.get(buf) is falsy:      yield from finalseg::cut(buf)   # §4
    else:                             yield each char of buf
```

**`cut_all_block(block)`** (0.42.1 includes English-run buffering — verify against source):

```
dag = get_DAG(block); old_j = -1; eng_scan = false; eng_buf = ""
for k in 0..N, L = dag[k]:
    if eng_scan and block[k] is not alnum: eng_scan = false; yield eng_buf
    if L.len() == 1 and k > old_j:
        word = block[k..=L[0]]
        if re_eng.match(word):
            if !eng_scan { eng_scan = true; eng_buf = word } else { eng_buf += word }
        if !eng_scan: yield word
        old_j = L[0]
    else:
        for j in L where j > k:
            yield block[k..=j]; old_j = j
if eng_scan: yield eng_buf
```

**`cut_for_search(sentence, HMM)`** — post-processes precise-mode words:

```
for w in cut(sentence, cut_all=false, HMM):
    if len(w) > 2:
        for each 2-char window g in w: if FREQ.get(g) > 0: yield g
    if len(w) > 3:
        for each 3-char window g in w: if FREQ.get(g) > 0: yield g
    yield w
```

## §4 finalseg (HMM for out-of-vocabulary runs)

Constants:
```
MIN_FLOAT = -3.14e100
states    = "BMES"
PrevStatus: B←{E,S}, M←{M,B}, S←{S,E}, E←{B,M}
re_han    = ([一-鿕]+)
re_skip   = ([a-zA-Z0-9]+(?:\.\d+)?%?)
Force_Split_Words: HashSet<String>   # populated by add_word(word, 0) / del_word
```

Tables `prob_start` (state → logp), `prob_trans` (state → state → logp), `prob_emit` (state → char → logp) come from `reference/jieba/jieba/finalseg/prob_*.py`, exported to JSON by `tools/gen_hmm_data.py`. Missing emission or transition → `MIN_FLOAT`.

Viterbi (`viterbi(obs, "BMES", start, trans, emit)`):
```
V[0][y] = start[y] + emit[y].get(obs[0], MIN_FLOAT)   for y in BMES; path[y]=[y]
for t in 1..len:
    for y in BMES:
        em = emit[y].get(obs[t], MIN_FLOAT)
        (prob, best_prev) = max over y0 in PrevStatus[y] of (V[t-1][y0] + trans[y0].get(y, MIN_FLOAT) + em, y0)
        # tie-break: Python max on (f64, char) → later char in PrevStatus string order wins
V_final: (prob, state) = max over y in "ES" of (V[last][y], y)    # tie → 'S' (S > E)
```
Decode B/M/E/S path to words: `B` starts a word, `E` ends it (yield), `S` yields the single char; trailing unclosed run is yielded whole.

Top-level `finalseg::cut(sentence)`:
```
for blk in re_han.split(sentence):
    if re_han.match(blk):
        for word in viterbi_cut(blk):
            if word in Force_Split_Words: yield each char
            else: yield word
    else:
        for x in re_skip.split(blk): if x nonempty: yield x
```

## §5 tokenize

- Input must be `str`; otherwise raise `ValueError("jieba: the input parameter should be unicode.")`.
- `mode="default"`: walk `cut(sentence, HMM=hmm)`, yield `(word, start, start+len)` where `len` is in **chars**; accumulate `start`.
- `mode="search"`: same walk, but before each word also yield in-dict 2-gram windows `(gram2, start+i, start+i+2)` then 3-gram windows, exactly like `cut_for_search`; then the word itself. Note order: all 2-grams, then all 3-grams, then the word.

## §6 Dictionary mutation

```
add_word(word, freq=None, tag=None):
    freq = int(freq) if given else suggest_freq(word, tune=False)
    FREQ[word] = freq
    total += freq                       # BUG-COMPATIBLE: old freq is NOT subtracted
    if tag: user_word_tag_tab[word] = tag
    insert 0-freq prefix markers for all prefixes of word not already present
    if freq == 0: finalseg Force_Split_Words.add(word)

del_word(word) = add_word(word, 0)

suggest_freq(segment, tune=False):
    ftotal = total as f64; freq = 1.0
    if segment is a str:
        word = segment
        for seg in cut(word, HMM=False): freq *= FREQ.get(seg, default=1) / ftotal
        freq = max( (freq * total) as int + 1, FREQ.get(word, default=1) )
    else:  # tuple of parts
        word = concat(parts)
        for seg in parts: freq *= FREQ.get(seg, default=1) / ftotal
        freq = min( (freq * total) as int, FREQ.get(word, default=0) )
    if tune: add_word(word, freq)
    return freq
```
Note the asymmetry: str branch uses `max(...)+1`-style raising (make the whole word win); tuple branch uses `min(...)` lowering (make the split win). `(x) as int` is Python `int()` = truncation toward zero.

```
load_userdict(f):   # f = path or file-like
    re_userdict = ^(.+?)( [0-9]+)?( [a-z]+)?$
    for each line: strip; skip empty; strip leading ﻿
    (word, freq, tag) = regex groups (freq/tag optional, stripped)
    add_word(word, freq, tag)           # freq None → suggest_freq path
```

## §7 posseg

Regexes:
```
re_han_detail   = ([一-鿕]+)
re_skip_detail  = ([\.0-9]+|[a-zA-Z0-9]+)
re_han_internal = ([一-鿕a-zA-Z0-9+#&\._]+)     # note: no % or - vs default cut
re_skip_internal= (\r\n|\s)
re_eng  = [a-zA-Z0-9]+ ;  re_num = [\.0-9]+ ;  re_eng1 = ^[a-zA-Z0-9]$
```

`word_tag_tab`: from the dict file, `word, _, tag = line.split(" ")` (entries without a tag would crash real jieba; the bundled dict always has 3 fields).

HMM tables: `posseg/prob_start.py`, `prob_trans.py`, `prob_emit.py` keyed by `(state, tag)` tuples like `('B','n')`; `char_state_tab.py` maps char → list of candidate `(state, tag)` states. `MIN_INF = -inf` for missing transitions in the candidate-pruning Viterbi:

```
viterbi(obs, char_state_tab, start, trans, emit):
    states0 = char_state_tab.get(obs[0], all_states)
    V[0][y] = start[y] + emit[y].get(obs[0], MIN_FLOAT); mem_path[0][y] = None
    for t in 1..len:
        prev = keys of mem_path[t-1] with non-empty trans[x]
        expected = union of trans[x].keys() for x in prev
        obs_states = (char_state_tab.get(obs[t], all_states) ∩ expected) or expected or all_states
        for y in obs_states:
            (prob, best) = max over y0 in prev of (V[t-1][y0] + trans[y0].get(y, MIN_INF) + emit[y].get(obs[t], MIN_FLOAT), y0)
    final: max over (V[last][y], y) for y in mem_path[last].keys(); backtrack
```
**Tie-breaking here compares `(f64, (state_char, tag))` tuples — Python orders the state tuples lexicographically (byte order of the strings). Replicate with an `Ord` impl comparing `state_char` then `tag`. Also `prev` iteration order affects `max` tie-breaks: Python iterates dict insertion order — preserve insertion order (use `indexmap` or a Vec) for exact parity.**

Structure mirrors `cut`: `cut_internal` splits on `re_han_internal`; han blocks → `cut_DAG` (HMM) or `cut_DAG_NO_HMM`; the DAG variants mirror §3 but yield `pair(word, tag)` with `word_tag_tab.get(word, 'x')`, and the HMM flush calls `cut_detail` (the posseg Viterbi) instead of finalseg. Non-han fallback tags: `\r\n|\s` → `x`; digits/dots → `m`; alnum → `eng`; else `x`. **Quirk to keep:** in `cut_internal`'s per-char fallback, the eng test is `re_eng.match(x)` on the whole fragment, not the single char.

`posseg.cut(HMM=False)` buffers alnum single chars (`re_eng1`) and yields the buffer as `pair(buf, 'eng')`.

`pair`: `.word`, `.flag`; `str(p) == f"{word}/{flag}"`; iterable as `(word, flag)`; eq/hash by tuple.

## §8 analyse

**TF-IDF (`extract_tags(sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False)`):**
```
words = posseg.cut(sentence) if allowPOS else cut(sentence)
freq counter over words, skipping: (allowPOS and flag not in allowPOS),
    len(word.strip()) < 2, word.lower() in STOP_WORDS
total = sum(freq.values())
weight[w] = freq[w] * idf.get(w, median_idf) / total
sort by weight DESC, stable (ties keep first-seen order); truncate to topK if topK
```
- `STOP_WORDS` is the hardcoded frozenset in `tfidf.py` (33 English words) — copy verbatim.
- `idf.txt`: lines `word idf_value`; `median_idf = sorted(values)[len/2]`.
- **Stable sort is mandatory** — Python's `sorted` is stable and dict preserves insertion order; use `sort_by` (stable) in Rust over first-seen-ordered entries.

**TextRank (`textrank(sentence, topK=20, withWeight=False, allowPOS=('ns','n','vn','v'))`):**
```
words = list(posseg.cut(sentence)); span = 5
keep(wp) = wp.flag in allowPOS and len(wp.word.strip()) >= 2 and wp.word.lower() not in STOP_WORDS
for i, wp in words where keep(wp):
    for j in (i+1)..(i+span) while j < len: if keep(words[j]): edge_weight[(wp.word, words[j].word)] += 1
graph: undirected (each edge added both directions)
rank: ws[n] = 1/len(graph) initially; 10 iterations of
      ws[n] = (1-0.85) + 0.85 * Σ_e (e.weight / outSum[e.target] * ws[e.target])
      **iterating nodes in sorted key order each pass** (jieba sorts for stability)
normalize: ws[n] = (w - min_rank/10) / (max_rank - min_rank/10)     # exact formula, quirky on purpose
sort like extract_tags
```

## §9 Module init & misc parity points

- **Lazy init:** nothing loads at `import jieba_rs`; first segmentation call triggers `initialize()`. `initialize(dictionary)` / `set_dictionary(path)` swap dicts before (or after) first load.
- `tmp_dir` and `cache_file` remain assignable compatibility attributes, but the
  Rust-backed initializer does not use jieba's marshal cache.
- `jieba_rs.get_FREQ(k, d=None)` = `FREQ.get(k, d)` — retained for compatibility.
- Logging: jieba writes "Building prefix dict..." etc. to a `logging` logger (`jieba.default_logger`) — replicate `setLogLevel` and emit similar init logs (content need not match byte-for-byte).
- `Tokenizer(dictionary=path)` accepts a custom main dictionary path.
