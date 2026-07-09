//! Prefix dictionary (`FREQ` + `total`), DAG construction, and the
//! max-probability route (`calc`). See docs/ALGORITHM_NOTES.md §1, §2.

use std::collections::HashMap;

const NO_TAG: u8 = 0;
const FREQ_BITS: u32 = 55;
const FREQ_MASK: u64 = (1_u64 << FREQ_BITS) - 1;
const OVERFLOW: u64 = 1_u64 << FREQ_BITS;
const TAG_SHIFT: u32 = 56;
const TAG_MASK: u64 = 0x7f_u64 << TAG_SHIFT;
const PRESENT: u64 = 1_u64 << 63;
const MIN_PACKED_FREQ: i64 = -(1_i64 << (FREQ_BITS - 1));
const MAX_PACKED_FREQ: i64 = (1_i64 << (FREQ_BITS - 1)) - 1;

/// One prefix-dictionary entry. POS tags are interned once per dictionary and
/// stored as compact IDs instead of duplicating every tagged word in a second
/// `HashMap<String, String>`.
#[derive(Debug, Clone, Copy)]
struct DictEntry(u64);
const _: () = assert!(std::mem::size_of::<DictEntry>() == 8);

impl DictEntry {
    #[inline(always)]
    fn new(freq: i64, tag: u8) -> Self {
        assert!(can_pack_freq(freq));
        Self(PRESENT | (u64::from(tag) << TAG_SHIFT) | (freq as u64 & FREQ_MASK))
    }

    #[inline(always)]
    fn is_present(self) -> bool {
        self.0 & PRESENT != 0
    }

    #[inline(always)]
    fn tag(self) -> u8 {
        ((self.0 & TAG_MASK) >> TAG_SHIFT) as u8
    }

    #[inline(always)]
    fn freq(self) -> i64 {
        ((self.0 & FREQ_MASK) << (64 - FREQ_BITS)) as i64 >> (64 - FREQ_BITS)
    }

    #[inline(always)]
    fn is_overflow(self) -> bool {
        self.0 & OVERFLOW != 0
    }

    #[inline(always)]
    fn set_freq(&mut self, freq: i64) {
        assert!(can_pack_freq(freq));
        self.0 = (self.0 & !FREQ_MASK) | (freq as u64 & FREQ_MASK);
    }

    #[inline(always)]
    fn set_overflow(&mut self, overflow: bool) {
        if overflow {
            self.0 |= OVERFLOW;
        } else {
            self.0 &= !OVERFLOW;
        }
    }

    #[inline(always)]
    fn set_present(&mut self, present: bool) {
        if present {
            self.0 |= PRESENT;
        } else {
            self.0 &= !PRESENT;
        }
    }
}

fn can_pack_freq(freq: i64) -> bool {
    (MIN_PACKED_FREQ..=MAX_PACKED_FREQ).contains(&freq)
}

/// The prefix dictionary: word/prefix -> frequency, plus the running total.
///
/// Every proper prefix of every word is present with freq 0 (a marker) so the
/// DAG scan can terminate correctly. `total` is the sum of real word freqs.
#[derive(Debug, Default, Clone)]
pub struct PrefixDict {
    entries: HashMap<String, DictEntry>,
    /// Frequencies outside the signed 55-bit common representation. This is
    /// empty for shipped dictionaries and normal user dictionaries, while
    /// preserving the full public `i64` mutation API without widening every
    /// hot dictionary entry.
    overflow_freq: HashMap<String, i64>,
    tags: Vec<Box<str>>,
    pub total: i64,
}

impl PrefixDict {
    /// Parse dict.txt-format content: lines of `word freq [tag]`.
    ///
    pub fn from_dict_str(content: &str) -> PrefixDict {
        let mut entries = HashMap::with_capacity(500_000);
        let mut overflow_freq = HashMap::new();
        let mut tags = Vec::with_capacity(64);
        let mut tag_ids: HashMap<&str, u8> = HashMap::with_capacity(64);
        let mut total: i64 = 0;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // jieba: word, freq = line.split(' ')[:2]
            let mut parts = line.split(' ');
            let word = match parts.next() {
                Some(w) if !w.is_empty() => w,
                _ => continue,
            };
            let f: i64 = match parts.next() {
                Some(f) => f.trim().parse().unwrap_or(0),
                None => continue,
            };
            let tag = parts
                .next()
                .filter(|tag| !tag.is_empty())
                .map(|tag| intern_tag(tag, &mut tags, &mut tag_ids))
                .unwrap_or(NO_TAG);
            insert_word_with_prefixes(&mut entries, &mut overflow_freq, word, f, tag);
            total += f;
        }

        PrefixDict {
            entries,
            overflow_freq,
            tags,
            total,
        }
    }

    /// `self.FREQ.get(word) or 1` — None or 0 both map to 1.
    #[inline]
    pub fn freq_or_one(&self, word: &str) -> i64 {
        let Some(entry) = self.entries.get(word) else {
            return 1;
        };
        if !entry.is_present() {
            return 1;
        }
        let freq = if entry.is_overflow() {
            self.overflow_freq[word]
        } else {
            entry.freq()
        };
        if freq == 0 {
            1
        } else {
            freq
        }
    }

    /// Truthy check `if self.FREQ.get(word):` — present and non-zero.
    #[inline]
    pub fn is_word(&self, word: &str) -> bool {
        let Some(entry) = self.entries.get(word) else {
            return false;
        };
        entry.is_present()
            && if entry.is_overflow() {
                self.overflow_freq[word] != 0
            } else {
                entry.0 & FREQ_MASK != 0
            }
    }

    /// Whether `word` (or a prefix marker) is present in FREQ at all.
    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.entries
            .get(word)
            .is_some_and(|entry| entry.is_present())
    }

    #[inline]
    pub fn get_freq(&self, word: &str) -> Option<i64> {
        self.entries
            .get(word)
            .filter(|entry| entry.is_present())
            .map(|entry| {
                if entry.is_overflow() {
                    self.overflow_freq[word]
                } else {
                    entry.freq()
                }
            })
    }

    #[inline]
    pub fn tag_of(&self, word: &str) -> Option<&str> {
        let tag = self
            .entries
            .get(word)
            .filter(|entry| entry.is_present())?
            .tag();
        if tag == NO_TAG {
            None
        } else {
            Some(&self.tags[usize::from(tag - 1)])
        }
    }

    pub fn set_freq(&mut self, word: &str, freq: i64) {
        let packed_freq = if can_pack_freq(freq) { freq } else { 0 };
        self.entries
            .entry(word.to_string())
            .and_modify(|entry| {
                entry.set_freq(packed_freq);
                entry.set_overflow(!can_pack_freq(freq));
                entry.set_present(true);
            })
            .or_insert_with(|| {
                let mut entry = DictEntry::new(packed_freq, NO_TAG);
                entry.set_overflow(!can_pack_freq(freq));
                entry
            });
        if can_pack_freq(freq) {
            self.overflow_freq.remove(word);
        } else {
            self.overflow_freq.insert(word.to_string(), freq);
        }
    }

    pub fn delete_freq(&mut self, word: &str) {
        if let Some(entry) = self.entries.get_mut(word) {
            entry.set_present(false);
        }
    }

    pub fn clear_freq(&mut self) {
        for entry in self.entries.values_mut() {
            entry.set_present(false);
        }
    }

    pub fn set_freq_entries<I: IntoIterator<Item = (String, i64)>>(&mut self, entries: I) {
        self.clear_freq();
        for (word, freq) in entries {
            self.set_freq(&word, freq);
        }
    }

    pub fn ensure_prefix(&mut self, prefix: &str) {
        self.entries
            .entry(prefix.to_string())
            .and_modify(|entry| {
                if !entry.is_present() {
                    entry.set_freq(0);
                    entry.set_present(true);
                }
            })
            .or_insert_with(|| DictEntry::new(0, NO_TAG));
    }
}

fn intern_tag<'a>(
    tag: &'a str,
    tags: &mut Vec<Box<str>>,
    tag_ids: &mut HashMap<&'a str, u8>,
) -> u8 {
    if let Some(&id) = tag_ids.get(tag) {
        return id;
    }
    let id = u8::try_from(tags.len() + 1).expect("dictionary has more than 127 POS tags");
    assert!(id <= 0x7f, "dictionary has more than 127 POS tags");
    tags.push(tag.into());
    tag_ids.insert(tag, id);
    id
}

/// Insert a word at `freq` and add 0-freq markers for all its proper prefixes.
fn insert_word_with_prefixes(
    entries: &mut HashMap<String, DictEntry>,
    overflow_freq: &mut HashMap<String, i64>,
    word: &str,
    freq: i64,
    tag: u8,
) {
    let packed_freq = if can_pack_freq(freq) { freq } else { 0 };
    let mut entry = DictEntry::new(packed_freq, tag);
    if !can_pack_freq(freq) {
        entry.set_overflow(true);
        overflow_freq.insert(word.to_string(), freq);
    }
    entries.insert(word.to_string(), entry);
    // prefixes by char (jieba: word[:ch+1] for ch in range(len(word)))
    let mut acc = String::new();
    let mut chars = word.chars().peekable();
    while let Some(c) = chars.next() {
        acc.push(c);
        if chars.peek().is_none() {
            // this is the whole word, already inserted above
            break;
        }
        entries
            .entry(acc.clone())
            .or_insert_with(|| DictEntry::new(0, NO_TAG));
    }
}

/// Build the DAG: for each start index k, the list of end indices i such that
/// chars[k..=i] is a real word. Terminates at the first fragment absent from
/// FREQ (prefix markers keep the scan alive).
pub fn get_dag(dict: &PrefixDict, chars: &[char]) -> Vec<Vec<usize>> {
    let n = chars.len();
    let mut dag: Vec<Vec<usize>> = Vec::with_capacity(n);
    for k in 0..n {
        let mut tmplist: Vec<usize> = Vec::new();
        let mut i = k;
        let mut frag = String::new();
        frag.push(chars[k]);
        while i < n && dict.contains(&frag) {
            if dict.is_word(&frag) {
                tmplist.push(i);
            }
            i += 1;
            if i < n {
                frag.push(chars[i]);
            }
        }
        if tmplist.is_empty() {
            tmplist.push(k);
        }
        dag.push(tmplist);
    }
    dag
}

/// Compute the max-probability route (right-to-left DP). `route[idx] = (logp, best_end)`.
/// Tie-break: on equal log-prob, the larger end index (longer word) wins, matching
/// Python `max` over `(value, x)` tuples.
pub fn calc(dict: &PrefixDict, chars: &[char], dag: &[Vec<usize>]) -> Vec<(f64, usize)> {
    let n = chars.len();
    let logtotal = (dict.total as f64).ln();
    let mut route: Vec<(f64, usize)> = vec![(0.0, 0); n + 1];
    route[n] = (0.0, 0);
    for idx in (0..n).rev() {
        let mut best: Option<(f64, usize)> = None;
        for &x in &dag[idx] {
            let word: String = chars[idx..=x].iter().collect();
            let v = (dict.freq_or_one(&word) as f64).ln() - logtotal + route[x + 1].0;
            match best {
                Some((bv, bx)) if !(v > bv || (v == bv && x > bx)) => {}
                _ => best = Some((v, x)),
            }
        }
        route[idx] = best.unwrap_or((0.0, idx));
    }
    route
}

#[cfg(test)]
mod tests {
    use super::PrefixDict;

    #[test]
    fn dictionary_parser_preserves_overflow_frequencies_and_tags() {
        let dict = PrefixDict::from_dict_str("极值词 9223372036854775807 nz\n");
        assert_eq!(dict.get_freq("极值词"), Some(i64::MAX));
        assert_eq!(dict.tag_of("极值词"), Some("nz"));
    }
}
