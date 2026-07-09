//! posseg: POS-tagging segmentation. See docs/ALGORITHM_NOTES.md §7.

use std::collections::HashMap;

use crate::dict::{calc, get_dag, PrefixDict};
use crate::hmm_data::{PossegModel, MIN_FLOAT};
use crate::regexes::*;

const MIN_INF: f64 = f64::NEG_INFINITY;
pub type TaggedToken = (String, String, bool);

struct PosCtx<'a> {
    dict: &'a PrefixDict,
    user_tag: &'a HashMap<String, String>,
    model: &'a PossegModel,
}

impl PosCtx<'_> {
    /// POS tag lookup: user dict overrides main dict; default 'x'.
    fn tag_of(&self, word: &str) -> String {
        if let Some(t) = self.user_tag.get(word) {
            t.clone()
        } else if let Some(t) = self.dict.tag_of(word) {
            t.to_string()
        } else {
            "x".to_string()
        }
    }

    // ---- posseg Viterbi over (state, tag) ----

    fn emit(&self, state: usize, ch: char) -> f64 {
        self.model.emit[state]
            .get(&ch)
            .copied()
            .unwrap_or(MIN_FLOAT)
    }

    fn viterbi(&self, obs: &[char]) -> Vec<usize> {
        let m = self.model;
        let n = obs.len();
        let state_count = m.states.len();
        assert!(
            state_count <= u16::MAX as usize,
            "posseg state table exceeds u16 backpointer capacity"
        );

        let init_states: &[usize] = m
            .char_state_tab
            .get(&obs[0])
            .map(Vec::as_slice)
            .unwrap_or(&m.all_states);
        let mut prev_scores = vec![MIN_FLOAT; state_count];
        let mut cur_scores = vec![MIN_FLOAT; state_count];
        let mut prev_keys: Vec<usize> = Vec::with_capacity(init_states.len());
        for &y in init_states {
            prev_scores[y] = m.start[y] + self.emit(y, obs[0]);
            prev_keys.push(y);
        }

        let mut back: Vec<u16> = vec![0; n.saturating_sub(1) * state_count];
        let mut seen = vec![0u32; state_count];
        let mut generation = 0u32;

        for (t, &ch) in obs.iter().enumerate().take(n).skip(1) {
            // prev_states = keys with non-empty trans
            let prev_states: Vec<usize> = prev_keys
                .iter()
                .copied()
                .filter(|&x| !m.trans_from[x].is_empty())
                .collect();
            // expected next states
            generation = generation.wrapping_add(1);
            if generation == 0 {
                seen.fill(0);
                generation = 1;
            }
            let mut expect: Vec<usize> = Vec::new();
            for &x in &prev_states {
                for &y in &m.trans_from[x] {
                    if seen[y] != generation {
                        seen[y] = generation;
                        expect.push(y);
                    }
                }
            }
            // obs_states = states.get(obs[t], all_states) ∩ expect
            let cand: &[usize] = m
                .char_state_tab
                .get(&ch)
                .map(Vec::as_slice)
                .unwrap_or(&m.all_states);
            let mut obs_states: Vec<usize> = cand
                .iter()
                .copied()
                .filter(|&y| seen[y] == generation)
                .collect();
            if obs_states.is_empty() {
                if !expect.is_empty() {
                    obs_states = expect;
                } else {
                    obs_states = m.all_states.clone();
                }
            }

            let back_offset = (t - 1) * state_count;
            for &y in &obs_states {
                let em = self.emit(y, ch);
                let mut best_prob = MIN_INF;
                let mut best_state = 0usize;
                let mut found = false;
                for &y0 in &prev_states {
                    let tr = m.trans[y0 * state_count + y];
                    let val = prev_scores[y0] + tr + em;
                    if !found || val > best_prob || (val == best_prob && y0 > best_state) {
                        best_prob = val;
                        best_state = y0;
                        found = true;
                    }
                }
                assert!(found, "posseg viterbi: no prev states");
                cur_scores[y] = best_prob;
                back[back_offset + y] = best_state as u16;
            }

            for &y in &prev_keys {
                prev_scores[y] = MIN_FLOAT;
            }
            std::mem::swap(&mut prev_scores, &mut cur_scores);
            prev_keys = obs_states;
        }

        // final: max over (V[-1][y], y)
        let mut best: Option<(f64, usize)> = None;
        for &y in &prev_keys {
            let val = prev_scores[y];
            match best {
                Some((bv, by)) if val < bv || (val == bv && y < by) => {}
                _ => best = Some((val, y)),
            }
        }
        let mut state = best.expect("posseg viterbi: empty final").1;

        let mut route: Vec<usize> = vec![0; n];
        for i in (0..n).rev() {
            route[i] = state;
            if i >= 1 {
                state = back[(i - 1) * state_count + state] as usize;
            }
        }
        route
    }

    fn viterbi_decode(&self, obs: &[char], out: &mut Vec<TaggedToken>) {
        if obs.is_empty() {
            return;
        }
        let route = self.viterbi(obs);
        let n = obs.len();
        let mut begin = 0usize;
        let mut nexti = 0usize;
        for i in 0..n {
            let st = &self.model.states[route[i]];
            match st.state {
                'B' => begin = i,
                'E' => {
                    out.push((obs[begin..=i].iter().collect(), st.tag.clone(), false));
                    nexti = i + 1;
                }
                'S' => {
                    out.push((obs[i].to_string(), st.tag.clone(), false));
                    nexti = i + 1;
                }
                _ => {}
            }
        }
        if nexti < n {
            let st = &self.model.states[route[nexti]];
            out.push((obs[nexti..].iter().collect(), st.tag.clone(), false));
        }
    }

    // ---- cut_detail: HMM path for OOV han buffers ----

    fn cut_detail(&self, sentence: &str, out: &mut Vec<TaggedToken>) {
        for seg in split_captured(&RE_HAN_DETAIL, sentence) {
            if seg.is_match {
                let chars: Vec<char> = seg.text.chars().collect();
                self.viterbi_decode(&chars, out);
            } else {
                for x in split_captured(&RE_SKIP_DETAIL, seg.text) {
                    if x.text.is_empty() {
                        continue;
                    }
                    let flag = if starts_num(x.text) {
                        "m"
                    } else if starts_alnum(x.text) {
                        "eng"
                    } else {
                        "x"
                    };
                    out.push((x.text.to_string(), flag.to_string(), false));
                }
            }
        }
    }

    // ---- DAG paths ----

    fn cut_dag_no_hmm(&self, chars: &[char], out: &mut Vec<TaggedToken>) {
        let dag = get_dag(self.dict, chars);
        let route = calc(self.dict, chars, &dag);
        let n = chars.len();
        let mut x = 0;
        let mut buf = String::new();
        while x < n {
            let y = route[x].1 + 1;
            let l_word: String = chars[x..y].iter().collect();
            if is_single_alnum(&l_word) {
                buf.push_str(&l_word);
            } else {
                if !buf.is_empty() {
                    out.push((std::mem::take(&mut buf), "eng".to_string(), false));
                }
                let tag = self.tag_of(&l_word);
                out.push((l_word, tag, true));
            }
            x = y;
        }
        if !buf.is_empty() {
            out.push((buf, "eng".to_string(), false));
        }
    }

    fn flush_pos(&self, buf: &str, out: &mut Vec<TaggedToken>) {
        let cnt = buf.chars().count();
        if cnt == 1 {
            out.push((buf.to_string(), self.tag_of(buf), true));
        } else if !self.dict.is_word(buf) {
            self.cut_detail(buf, out);
        } else {
            for c in buf.chars() {
                let cs = c.to_string();
                let tag = self.tag_of(&cs);
                out.push((cs, tag, true));
            }
        }
    }

    fn cut_dag_hmm(&self, chars: &[char], out: &mut Vec<TaggedToken>) {
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
                    self.flush_pos(&b, out);
                }
                let l_word: String = chars[x..y].iter().collect();
                let tag = self.tag_of(&l_word);
                out.push((l_word, tag, true));
            }
            x = y;
        }
        if !buf.is_empty() {
            self.flush_pos(&buf, out);
        }
    }

    fn cut_internal(&self, sentence: &str, hmm: bool, out: &mut Vec<TaggedToken>) {
        for seg in split_captured(&RE_HAN_INTERNAL, sentence) {
            if seg.is_match {
                let chars: Vec<char> = seg.text.chars().collect();
                if hmm {
                    self.cut_dag_hmm(&chars, out);
                } else {
                    self.cut_dag_no_hmm(&chars, out);
                }
            } else {
                for x in split_captured(&RE_SKIP_INTERNAL, seg.text) {
                    if x.is_match {
                        out.push((x.text.to_string(), "x".to_string(), false));
                    } else {
                        let x_alnum = starts_alnum(x.text);
                        for xx in x.text.chars() {
                            let xs = xx.to_string();
                            let flag = if starts_num(&xs) {
                                "m"
                            } else if x_alnum {
                                "eng"
                            } else {
                                "x"
                            };
                            out.push((xs, flag.to_string(), false));
                        }
                    }
                }
            }
        }
    }
}

/// posseg.cut(sentence, HMM) -> list of (word, flag).
#[allow(clippy::too_many_arguments)]
pub fn cut(
    dict: &PrefixDict,
    user_tag: &HashMap<String, String>,
    model: &PossegModel,
    sentence: &str,
    hmm: bool,
) -> Vec<(String, String)> {
    cut_detailed(dict, user_tag, model, sentence, hmm)
        .into_iter()
        .map(|(word, flag, _)| (word, flag))
        .collect()
}

/// POS cut with a marker for tags obtained from the dictionary tables.
#[allow(clippy::too_many_arguments)]
pub fn cut_detailed(
    dict: &PrefixDict,
    user_tag: &HashMap<String, String>,
    model: &PossegModel,
    sentence: &str,
    hmm: bool,
) -> Vec<TaggedToken> {
    let ctx = PosCtx {
        dict,
        user_tag,
        model,
    };
    let mut out = Vec::new();
    ctx.cut_internal(sentence, hmm, &mut out);
    out
}
