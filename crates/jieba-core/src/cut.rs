//! The four cut paths (full / precise+HMM / precise / for-search) and tokenize.
//! See docs/ALGORITHM_NOTES.md §3, §5.

use std::collections::HashSet;

use crate::dict::{calc, get_dag, PrefixDict};
use crate::hmm;
use crate::hmm_data::FinalsegModel;
use crate::regexes::*;

/// Context the cut paths need: the prefix dict, finalseg HMM tables, and the
/// force-split word set (words added with freq 0).
pub struct CutCtx<'a> {
    pub dict: &'a PrefixDict,
    /// Present only when an HMM path may run; the no-HMM paths never touch it.
    pub finalseg: Option<&'a FinalsegModel>,
    pub force_split: &'a HashSet<String>,
}

impl CutCtx<'_> {
    // ---- block functions ----

    fn cut_dag_no_hmm(&self, chars: &[char], out: &mut Vec<String>) {
        let dag = get_dag(self.dict, chars);
        let route = calc(self.dict, chars, &dag);
        let n = chars.len();
        let mut x = 0;
        let mut buf = String::new();
        while x < n {
            let y = route[x].1 + 1;
            // single alnum char -> buffer
            if y - x == 1 && chars[x].is_ascii_alphanumeric() {
                buf.push(chars[x]);
            } else {
                if !buf.is_empty() {
                    out.push(std::mem::take(&mut buf));
                }
                out.push(chars[x..y].iter().collect());
            }
            x = y;
        }
        if !buf.is_empty() {
            out.push(buf);
        }
    }

    fn flush_hmm(&self, buf: &str, out: &mut Vec<String>) {
        let cnt = buf.chars().count();
        if cnt == 1 {
            out.push(buf.to_string());
        } else if !self.dict.is_word(buf) {
            let model = self.finalseg.expect("finalseg model required for HMM cut");
            hmm::cut(buf, model, self.force_split, out);
        } else {
            for c in buf.chars() {
                out.push(c.to_string());
            }
        }
    }

    fn cut_dag_hmm(&self, chars: &[char], out: &mut Vec<String>) {
        let dag = get_dag(self.dict, chars);
        let route = calc(self.dict, chars, &dag);
        let n = chars.len();
        let mut x = 0;
        let mut buf = String::new();
        while x < n {
            let y = route[x].1 + 1;
            if y - x == 1 {
                buf.push(chars[x]);
            } else {
                if !buf.is_empty() {
                    let b = std::mem::take(&mut buf);
                    self.flush_hmm(&b, out);
                }
                out.push(chars[x..y].iter().collect());
            }
            x = y;
        }
        if !buf.is_empty() {
            self.flush_hmm(&buf, out);
        }
    }

    fn cut_all_block(&self, chars: &[char], out: &mut Vec<String>) {
        let n = chars.len();
        let mut old_j: i64 = -1;
        let mut eng_scan = false;
        let mut eng_buf = String::new();
        let mut ends: Vec<usize> = Vec::new();
        for k in 0..n {
            ends.clear();
            let mut i = k;
            let mut frag = String::new();
            frag.push(chars[k]);
            while i < n && self.dict.contains(&frag) {
                if self.dict.is_word(&frag) {
                    ends.push(i);
                }
                i += 1;
                if i < n {
                    frag.push(chars[i]);
                }
            }
            if ends.is_empty() {
                ends.push(k);
            }

            if eng_scan && !chars[k].is_ascii_alphanumeric() {
                eng_scan = false;
                out.push(std::mem::take(&mut eng_buf));
            }
            if ends.len() == 1 && (k as i64) > old_j {
                let j0 = ends[0];
                let word: String = chars[k..=j0].iter().collect();
                if starts_alnum(&word) {
                    if !eng_scan {
                        eng_scan = true;
                        eng_buf.push_str(&word);
                    } else {
                        eng_buf.push_str(&word);
                    }
                }
                if !eng_scan {
                    out.push(word);
                }
                old_j = j0 as i64;
            } else {
                for &j in &ends {
                    if j > k {
                        out.push(chars[k..=j].iter().collect());
                        old_j = j as i64;
                    }
                }
            }
        }
        if eng_scan {
            out.push(eng_buf);
        }
    }

    // ---- top-level cut ----

    pub fn cut(&self, sentence: &str, cut_all: bool, hmm: bool) -> Vec<String> {
        let mut out = Vec::new();
        // jieba uses re_han_default / re_skip_default for ALL modes; cut_all only
        // selects the block function and the non-han else-branch behavior.
        for seg in split_captured(&RE_HAN_DEFAULT, sentence) {
            let blk = seg.text;
            if blk.is_empty() {
                continue;
            }
            if seg.is_match {
                let chars: Vec<char> = blk.chars().collect();
                if cut_all {
                    self.cut_all_block(&chars, &mut out);
                } else if hmm {
                    self.cut_dag_hmm(&chars, &mut out);
                } else {
                    self.cut_dag_no_hmm(&chars, &mut out);
                }
            } else {
                for x in split_captured(&RE_SKIP_DEFAULT, blk) {
                    if x.is_match {
                        out.push(x.text.to_string());
                    } else if !cut_all {
                        // per-char (empty pieces yield nothing)
                        for c in x.text.chars() {
                            out.push(c.to_string());
                        }
                    } else {
                        // cut_all: yield the whole piece (including empty strings)
                        out.push(x.text.to_string());
                    }
                }
            }
        }
        out
    }

    pub fn cut_for_search(&self, sentence: &str, hmm: bool) -> Vec<String> {
        let words = self.cut(sentence, false, hmm);
        let mut out = Vec::with_capacity(words.len());
        for w in words {
            let wc: Vec<char> = w.chars().collect();
            let l = wc.len();
            if l > 2 {
                for i in 0..l - 1 {
                    let gram2: String = wc[i..i + 2].iter().collect();
                    if self.dict.is_word(&gram2) {
                        out.push(gram2);
                    }
                }
            }
            if l > 3 {
                for i in 0..l - 2 {
                    let gram3: String = wc[i..i + 3].iter().collect();
                    if self.dict.is_word(&gram3) {
                        out.push(gram3);
                    }
                }
            }
            out.push(w);
        }
        out
    }

    pub fn tokenize(&self, sentence: &str, search: bool, hmm: bool) -> Vec<(String, usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0usize;
        for w in self.cut(sentence, false, hmm) {
            let wc: Vec<char> = w.chars().collect();
            let width = wc.len();
            if search {
                if width > 2 {
                    for i in 0..width - 1 {
                        let gram2: String = wc[i..i + 2].iter().collect();
                        if self.dict.is_word(&gram2) {
                            out.push((gram2, start + i, start + i + 2));
                        }
                    }
                }
                if width > 3 {
                    for i in 0..width - 2 {
                        let gram3: String = wc[i..i + 3].iter().collect();
                        if self.dict.is_word(&gram3) {
                            out.push((gram3, start + i, start + i + 3));
                        }
                    }
                }
            }
            out.push((w, start, start + width));
            start += width;
        }
        out
    }
}
