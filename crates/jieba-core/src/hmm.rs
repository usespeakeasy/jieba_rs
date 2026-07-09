//! finalseg: the HMM (Viterbi) segmentation of out-of-vocabulary runs.
//! See docs/ALGORITHM_NOTES.md §4.

use std::collections::HashSet;

use crate::hmm_data::{FinalsegModel, MIN_FLOAT};
use crate::regexes::{matches_at_start, split_captured, RE_HAN_FINALSEG, RE_SKIP_FINALSEG};

const STATES: [char; 4] = ['B', 'M', 'E', 'S'];

#[inline]
fn st_idx(c: char) -> usize {
    match c {
        'B' => 0,
        'M' => 1,
        'E' => 2,
        'S' => 3,
        _ => unreachable!(),
    }
}

/// PrevStatus: allowed predecessor states for each state.
#[inline]
fn prev_status(y: char) -> &'static [char] {
    match y {
        'B' => &['E', 'S'],
        'M' => &['M', 'B'],
        'S' => &['S', 'E'],
        'E' => &['B', 'M'],
        _ => unreachable!(),
    }
}

/// `max` over candidates with Python tuple tie-break `(value, char)`: on equal
/// value the larger char wins. Returns the winning `(value, char)`.
#[inline]
fn best_by_char(cands: impl Iterator<Item = (f64, char)>) -> (f64, char) {
    let mut best: Option<(f64, char)> = None;
    for (v, c) in cands {
        match best {
            Some((bv, bc)) if v < bv || (v == bv && c < bc) => {}
            _ => best = Some((v, c)),
        }
    }
    best.expect("no candidates")
}

fn viterbi(obs: &[char], m: &FinalsegModel) -> Vec<char> {
    let n = obs.len();
    let emit = |y: char, ch: char| -> f64 {
        m.emit
            .get(&y)
            .and_then(|e| e.get(&ch))
            .copied()
            .unwrap_or(MIN_FLOAT)
    };
    let start = |y: char| -> f64 { m.start.get(&y).copied().unwrap_or(MIN_FLOAT) };
    let trans = |y0: char, y: char| -> f64 {
        m.trans
            .get(&y0)
            .and_then(|t| t.get(&y))
            .copied()
            .unwrap_or(MIN_FLOAT)
    };

    let mut v_prev = [0.0f64; 4];
    for &y in &STATES {
        v_prev[st_idx(y)] = start(y) + emit(y, obs[0]);
    }
    // bp[t][state_idx] = best previous state char
    let mut bp: Vec<[char; 4]> = Vec::with_capacity(n);
    bp.push(['\0'; 4]);

    for &ch in obs.iter().take(n).skip(1) {
        let mut v_cur = [MIN_FLOAT; 4];
        let mut ptr = ['\0'; 4];
        for &y in &STATES {
            let em = emit(y, ch);
            let (prob, state) = best_by_char(
                prev_status(y)
                    .iter()
                    .map(|&y0| (v_prev[st_idx(y0)] + trans(y0, y) + em, y0)),
            );
            v_cur[st_idx(y)] = prob;
            ptr[st_idx(y)] = state;
        }
        v_prev = v_cur;
        bp.push(ptr);
    }

    // final state: max over ('E','S')
    let (_p, mut state) = best_by_char(['E', 'S'].iter().map(|&y| (v_prev[st_idx(y)], y)));

    let mut route = vec!['\0'; n];
    for t in (0..n).rev() {
        route[t] = state;
        if t > 0 {
            state = bp[t][st_idx(state)];
        }
    }
    route
}

/// Decode a B/M/E/S route into words (jieba `finalseg.__cut`).
fn decode(obs: &[char], route: &[char], out: &mut Vec<String>) {
    let n = obs.len();
    let mut begin = 0usize;
    let mut nexti = 0usize;
    for i in 0..n {
        match route[i] {
            'B' => begin = i,
            'E' => {
                out.push(obs[begin..=i].iter().collect());
                nexti = i + 1;
            }
            'S' => {
                out.push(obs[i].to_string());
                nexti = i + 1;
            }
            _ => {} // 'M'
        }
    }
    if nexti < n {
        out.push(obs[nexti..].iter().collect());
    }
}

fn viterbi_cut(block: &[char], m: &FinalsegModel, out: &mut Vec<String>) {
    if block.is_empty() {
        return;
    }
    let route = viterbi(block, m);
    decode(block, &route, out);
}

/// finalseg.cut(sentence): split into han / non-han blocks, HMM-segment han,
/// apply force-split. Operates on a &str buffer (may contain non-han chars).
pub fn cut(
    sentence: &str,
    m: &FinalsegModel,
    force_split: &HashSet<String>,
    out: &mut Vec<String>,
) {
    for seg in split_captured(&RE_HAN_FINALSEG, sentence) {
        if matches_at_start(&RE_HAN_FINALSEG, seg.text) {
            let chars: Vec<char> = seg.text.chars().collect();
            let mut words: Vec<String> = Vec::new();
            viterbi_cut(&chars, m, &mut words);
            for w in words {
                if force_split.contains(&w) {
                    for c in w.chars() {
                        out.push(c.to_string());
                    }
                } else {
                    out.push(w);
                }
            }
        } else {
            for x in split_captured(&RE_SKIP_FINALSEG, seg.text) {
                if !x.text.is_empty() {
                    out.push(x.text.to_string());
                }
            }
        }
    }
}
