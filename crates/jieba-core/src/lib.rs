//! jieba-core: a pure-Rust reimplementation of jieba 0.42.1's segmentation
//! algorithms, built for byte-exact parity. See docs/ALGORITHM_NOTES.md.

mod analyse;
mod cut;
mod dict;
mod hmm;
mod hmm_data;
mod posseg;
mod regexes;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cut::CutCtx;
use dict::PrefixDict;
use hmm_data::HmmData;

pub use analyse::STOP_WORDS;

/// A segment expected by `suggest_freq`: either a whole word, or the parts it
/// should be split into.
pub enum Segment {
    Word(String),
    Parts(Vec<String>),
}

/// IDF table for TF-IDF keyword extraction.
struct IdfData {
    idf_freq: HashMap<String, f64>,
    median_idf: f64,
}

impl IdfData {
    fn from_str(content: &str) -> IdfData {
        let mut idf_freq = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // jieba: word, freq = line.strip().split(' ')
            let mut it = line.split(' ');
            if let (Some(w), Some(f)) = (it.next(), it.next()) {
                if let Ok(v) = f.parse::<f64>() {
                    idf_freq.insert(w.to_string(), v);
                }
            }
        }
        let mut vals: Vec<f64> = idf_freq.values().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idf = if vals.is_empty() {
            0.0
        } else {
            vals[vals.len() / 2]
        };
        IdfData {
            idf_freq,
            median_idf,
        }
    }
}

/// The jieba Tokenizer. Owns its dictionary, so multiple instances are fully
/// independent (no global mutable state).
pub struct Tokenizer {
    dict_path: PathBuf,
    hmm_path: PathBuf,
    idf_path: PathBuf,

    dict: Option<PrefixDict>,
    user_word_tag_tab: HashMap<String, String>,
    force_split: HashSet<String>,

    hmm: Option<Arc<HmmData>>,
    idf: Option<Arc<IdfData>>,
    /// Extra stop words added via `set_stop_words` (union with the default set).
    stop_words_extra: HashSet<String>,
}

impl Tokenizer {
    /// Create a tokenizer that loads data files lazily from the given paths.
    pub fn new(
        dict_path: impl AsRef<Path>,
        hmm_path: impl AsRef<Path>,
        idf_path: impl AsRef<Path>,
    ) -> Tokenizer {
        Tokenizer {
            dict_path: dict_path.as_ref().to_path_buf(),
            hmm_path: hmm_path.as_ref().to_path_buf(),
            idf_path: idf_path.as_ref().to_path_buf(),
            dict: None,
            user_word_tag_tab: HashMap::new(),
            force_split: HashSet::new(),
            hmm: None,
            idf: None,
            stop_words_extra: HashSet::new(),
        }
    }

    /// Point at a different IDF corpus (for TF-IDF); forces reload on next use.
    pub fn set_idf_path(&mut self, idf_path: impl AsRef<Path>) {
        self.idf_path = idf_path.as_ref().to_path_buf();
        self.idf = None;
    }

    /// Add stop words (union with the hardcoded default set).
    pub fn add_stop_words<I: IntoIterator<Item = String>>(&mut self, words: I) {
        self.stop_words_extra.extend(words);
    }

    /// Point at a different main dictionary; forces a reload on next use.
    pub fn set_dictionary(&mut self, dict_path: impl AsRef<Path>) {
        self.dict_path = dict_path.as_ref().to_path_buf();
        self.dict = None;
    }

    pub fn is_initialized(&self) -> bool {
        self.dict.is_some()
    }

    /// Load the prefix dict + POS tag table (idempotent).
    pub fn initialize(&mut self) -> std::io::Result<()> {
        if self.dict.is_some() {
            return Ok(());
        }
        let content = std::fs::read_to_string(&self.dict_path)?;
        self.dict = Some(PrefixDict::from_dict_str(&content));
        Ok(())
    }

    fn ensure_dict(&mut self) {
        if self.dict.is_none() {
            self.initialize().expect("failed to load dictionary");
        }
    }

    fn ensure_hmm(&mut self) -> Arc<HmmData> {
        if let Some(h) = &self.hmm {
            return h.clone();
        }
        let content = std::fs::read_to_string(&self.hmm_path).expect("failed to read hmm data");
        let data = Arc::new(HmmData::from_json_str(&content).expect("failed to parse hmm data"));
        self.hmm = Some(data.clone());
        data
    }

    fn ensure_idf(&mut self) -> Arc<IdfData> {
        if let Some(i) = &self.idf {
            return i.clone();
        }
        let content = std::fs::read_to_string(&self.idf_path).expect("failed to read idf data");
        let data = Arc::new(IdfData::from_str(&content));
        self.idf = Some(data.clone());
        data
    }

    fn dict_ref(&self) -> &PrefixDict {
        self.dict.as_ref().expect("dict not initialized")
    }

    // ---- segmentation ----

    pub fn cut(&mut self, sentence: &str, cut_all: bool, hmm: bool) -> Vec<String> {
        self.ensure_dict();
        let model = if hmm && !cut_all {
            Some(self.ensure_hmm())
        } else {
            None
        };
        let ctx = CutCtx {
            dict: self.dict_ref(),
            finalseg: model.as_ref().map(|m| &m.finalseg),
            force_split: &self.force_split,
        };
        ctx.cut(sentence, cut_all, hmm)
    }

    pub fn cut_for_search(&mut self, sentence: &str, hmm: bool) -> Vec<String> {
        self.ensure_dict();
        let model = if hmm { Some(self.ensure_hmm()) } else { None };
        let ctx = CutCtx {
            dict: self.dict_ref(),
            finalseg: model.as_ref().map(|m| &m.finalseg),
            force_split: &self.force_split,
        };
        ctx.cut_for_search(sentence, hmm)
    }

    pub fn tokenize(
        &mut self,
        sentence: &str,
        search: bool,
        hmm: bool,
    ) -> Vec<(String, usize, usize)> {
        self.ensure_dict();
        let model = if hmm { Some(self.ensure_hmm()) } else { None };
        let ctx = CutCtx {
            dict: self.dict_ref(),
            finalseg: model.as_ref().map(|m| &m.finalseg),
            force_split: &self.force_split,
        };
        ctx.tokenize(sentence, search, hmm)
    }

    /// Build the DAG (end indices per start index) — for API-surface parity.
    pub fn get_dag(&mut self, sentence: &str) -> Vec<Vec<usize>> {
        self.ensure_dict();
        let chars: Vec<char> = sentence.chars().collect();
        dict::get_dag(self.dict_ref(), &chars)
    }

    /// Max-probability route for a given DAG — for API-surface parity.
    pub fn calc(&mut self, sentence: &str, dag: &[Vec<usize>]) -> Vec<(f64, usize)> {
        self.ensure_dict();
        let chars: Vec<char> = sentence.chars().collect();
        dict::calc(self.dict_ref(), &chars, dag)
    }

    pub fn pos_cut(&mut self, sentence: &str, hmm: bool) -> Vec<(String, String)> {
        self.ensure_dict();
        let model = self.ensure_hmm();
        posseg::cut(
            self.dict_ref(),
            &self.user_word_tag_tab,
            &model.posseg,
            sentence,
            hmm,
        )
    }

    pub fn pos_cut_detailed(&mut self, sentence: &str, hmm: bool) -> Vec<(String, String, bool)> {
        self.ensure_dict();
        let model = self.ensure_hmm();
        posseg::cut_detailed(
            self.dict_ref(),
            &self.user_word_tag_tab,
            &model.posseg,
            sentence,
            hmm,
        )
    }

    // ---- dictionary queries ----

    pub fn get_freq(&mut self, word: &str) -> Option<i64> {
        self.ensure_dict();
        self.dict_ref().get_freq(word)
    }

    /// Set one raw FREQ entry without changing `total` or adding prefixes.
    /// This mirrors direct mutation of jieba's public `Tokenizer.FREQ` dict.
    pub fn set_freq(&mut self, word: &str, freq: i64) {
        self.ensure_dict();
        self.dict.as_mut().unwrap().set_freq(word, freq);
    }

    /// Delete one raw FREQ entry without changing `total`.
    pub fn delete_freq(&mut self, word: &str) {
        self.ensure_dict();
        self.dict.as_mut().unwrap().delete_freq(word);
    }

    /// Clear the raw FREQ table without changing `total`.
    pub fn clear_freq(&mut self) {
        self.ensure_dict();
        self.dict.as_mut().unwrap().clear_freq();
    }

    /// Replace the raw FREQ table without changing `total`.
    pub fn set_freq_entries<I: IntoIterator<Item = (String, i64)>>(&mut self, entries: I) {
        self.ensure_dict();
        self.dict.as_mut().unwrap().set_freq_entries(entries);
    }

    pub fn total(&mut self) -> i64 {
        self.ensure_dict();
        self.dict_ref().total
    }

    /// Replace the probability denominator exposed as `Tokenizer.total`.
    pub fn set_total(&mut self, total: i64) {
        self.ensure_dict();
        self.dict.as_mut().unwrap().total = total;
    }

    /// Add a word to finalseg's force-split set without touching FREQ.
    pub fn add_force_split(&mut self, word: &str) {
        self.force_split.insert(word.to_string());
    }

    pub fn set_force_split_words<I: IntoIterator<Item = String>>(&mut self, words: I) {
        self.force_split.clear();
        self.force_split.extend(words);
    }

    // ---- mutation ----

    pub fn add_word(&mut self, word: &str, freq: Option<i64>, tag: Option<&str>) {
        self.ensure_dict();
        let f = freq.unwrap_or_else(|| self.suggest_freq(&Segment::Word(word.to_string()), false));
        let dict = self.dict.as_mut().unwrap();
        dict.set_freq(word, f);
        dict.total += f;
        insert_word_with_prefixes_only_markers(dict, word);
        if let Some(t) = tag {
            if !t.is_empty() {
                self.user_word_tag_tab
                    .insert(word.to_string(), t.to_string());
            }
        }
        if f == 0 {
            self.force_split.insert(word.to_string());
        }
    }

    pub fn del_word(&mut self, word: &str) {
        self.add_word(word, Some(0), None);
    }

    pub fn suggest_freq(&mut self, segment: &Segment, tune: bool) -> i64 {
        self.ensure_dict();
        let total = self.dict_ref().total;
        let ftotal = total as f64;
        let (result, word) = match segment {
            Segment::Word(word) => {
                let mut freq = 1.0f64;
                for seg in self.cut(word, false, false) {
                    let g = self.dict_ref().get_freq(&seg).unwrap_or(1);
                    freq *= g as f64 / ftotal;
                }
                let computed = (freq * total as f64) as i64 + 1;
                let existing = self.dict_ref().get_freq(word).unwrap_or(1);
                (computed.max(existing), word.clone())
            }
            Segment::Parts(parts) => {
                let word: String = parts.concat();
                let mut freq = 1.0f64;
                for seg in parts {
                    let g = self.dict_ref().get_freq(seg).unwrap_or(1);
                    freq *= g as f64 / ftotal;
                }
                let computed = (freq * total as f64) as i64;
                let existing = self.dict_ref().get_freq(&word).unwrap_or(0);
                (computed.min(existing), word)
            }
        };
        if tune {
            self.add_word(&word, Some(result), None);
        }
        result
    }

    /// Load a user dictionary from text content (parsing per jieba's re_userdict).
    pub fn load_userdict_str(&mut self, content: &str) {
        self.ensure_dict();
        for raw in content.lines() {
            let line = raw.trim().trim_start_matches('\u{feff}');
            if line.is_empty() {
                continue;
            }
            let (word, freq, tag) = parse_userdict_line(line);
            let freq_val = freq.map(|f| f.trim().parse::<i64>().unwrap_or(0));
            self.add_word(word, freq_val, tag.map(|t| t.trim()));
        }
    }

    // ---- analyse ----

    pub fn extract_tags(
        &mut self,
        sentence: &str,
        top_k: Option<usize>,
        allow_pos: &[String],
    ) -> Vec<(String, f64)> {
        self.ensure_dict();
        let idf = self.ensure_idf();
        let words: Vec<(String, Option<String>)> = if allow_pos.is_empty() {
            self.cut(sentence, false, true)
                .into_iter()
                .map(|w| (w, None))
                .collect()
        } else {
            self.pos_cut(sentence, true)
                .into_iter()
                .map(|(w, f)| (w, Some(f)))
                .collect()
        };
        analyse::extract_tags(
            words,
            allow_pos,
            &idf.idf_freq,
            idf.median_idf,
            top_k,
            &self.stop_words_extra,
        )
    }

    pub fn textrank(
        &mut self,
        sentence: &str,
        top_k: Option<usize>,
        allow_pos: &[String],
    ) -> Vec<(String, f64)> {
        self.ensure_dict();
        self.ensure_hmm();
        let words = self.pos_cut(sentence, true);
        let extra = std::mem::take(&mut self.stop_words_extra);
        let out = analyse::textrank(words, allow_pos, top_k, &extra);
        self.stop_words_extra = extra;
        out
    }
}

/// Add only the 0-freq prefix markers (the word itself was already inserted).
fn insert_word_with_prefixes_only_markers(dict: &mut PrefixDict, word: &str) {
    let mut acc = String::new();
    let mut chars = word.chars().peekable();
    while let Some(c) = chars.next() {
        acc.push(c);
        if chars.peek().is_none() {
            break;
        }
        dict.ensure_prefix(&acc);
    }
}

/// jieba re_userdict = `^(.+?)( [0-9]+)?( [a-z]+)?$` (non-greedy word, optional
/// space-prefixed freq, optional space-prefixed tag). Returns (word, freq, tag)
/// with the leading spaces stripped from freq/tag groups.
fn parse_userdict_line(line: &str) -> (&str, Option<&str>, Option<&str>) {
    // Match from the end: optional " [a-z]+" tag, then optional " [0-9]+" freq.
    let mut rest = line;
    let mut tag: Option<&str> = None;
    let mut freq: Option<&str> = None;

    // tag: trailing " [a-z]+"
    if let Some(pos) = rest.rfind(' ') {
        let cand = &rest[pos + 1..];
        if !cand.is_empty() && cand.chars().all(|c| c.is_ascii_lowercase()) {
            // Ensure there is a non-empty word before it (non-greedy .+? needs >=1 char)
            if pos >= 1 {
                tag = Some(cand);
                rest = &rest[..pos];
            }
        }
    }
    // freq: trailing " [0-9]+"
    if let Some(pos) = rest.rfind(' ') {
        let cand = &rest[pos + 1..];
        if !cand.is_empty() && cand.chars().all(|c| c.is_ascii_digit()) && pos >= 1 {
            freq = Some(cand);
            rest = &rest[..pos];
        }
    }
    (rest, freq, tag)
}

pub fn ping() -> &'static str {
    "pong"
}

#[cfg(test)]
mod tests;
