//! Loading of the HMM probability tables produced by `tools/gen_hmm_data.py`
//! (see docs/ALGORITHM_NOTES.md §4, §7). The JSON encodes posseg (state, tag)
//! tuple keys as a single string `state + tag`, e.g. ("B","nr") -> "Bnr".

use serde::Deserialize;
use std::collections::HashMap;

/// The very negative log-probability jieba uses for missing emissions.
pub const MIN_FLOAT: f64 = -3.14e100;

/// A posseg state: an HMM position tag (B/M/E/S) paired with a POS tag.
///
/// `Ord` derives lexicographically (state char first, then tag string), which
/// matches Python's tuple comparison used for `max` tie-breaking in the posseg
/// Viterbi decode.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateTag {
    pub state: char,
    pub tag: String,
}

impl StateTag {
    /// Decode an encoded "state+tag" key: first char is the state, rest is tag.
    fn decode(s: &str) -> StateTag {
        let mut chars = s.chars();
        let state = chars.next().expect("empty state-tag key");
        let tag = chars.as_str().to_string();
        StateTag { state, tag }
    }
}

/// finalseg HMM tables, keyed by the plain state char.
#[derive(Debug)]
pub struct FinalsegModel {
    pub start: HashMap<char, f64>,
    pub trans: HashMap<char, HashMap<char, f64>>,
    pub emit: HashMap<char, HashMap<char, f64>>,
}

/// posseg HMM tables, keyed by `StateTag`.
#[derive(Debug)]
pub struct PossegModel {
    pub states: Vec<StateTag>,
    pub start: Vec<f64>,
    /// Dense transition table indexed by `from * states.len() + to`.
    pub trans: Vec<f64>,
    /// Non-empty transition targets per source state.
    pub trans_from: Vec<Vec<usize>>,
    pub emit: Vec<HashMap<char, f64>>,
    /// char -> candidate state IDs.
    pub char_state_tab: HashMap<char, Vec<usize>>,
    /// `trans_p.keys()` — the fallback "all states" set.
    pub all_states: Vec<usize>,
}

#[derive(Debug)]
pub struct HmmData {
    pub finalseg: FinalsegModel,
    pub posseg: PossegModel,
}

// ---- raw serde shapes matching the JSON ----

#[derive(Deserialize)]
struct RawFinalseg {
    start: HashMap<String, f64>,
    trans: HashMap<String, HashMap<String, f64>>,
    emit: HashMap<String, HashMap<String, f64>>,
}

#[derive(Deserialize)]
struct RawPosseg {
    start: HashMap<String, f64>,
    trans: HashMap<String, HashMap<String, f64>>,
    emit: HashMap<String, HashMap<String, f64>>,
    char_state_tab: HashMap<String, Vec<String>>,
}

#[derive(Deserialize)]
struct RawHmm {
    finalseg: RawFinalseg,
    posseg: RawPosseg,
}

fn one_char(s: &str) -> char {
    let mut it = s.chars();
    let c = it.next().expect("empty char key");
    debug_assert!(it.next().is_none(), "expected single-char key, got {s:?}");
    c
}

fn char_map(m: HashMap<String, f64>) -> HashMap<char, f64> {
    m.into_iter().map(|(k, v)| (one_char(&k), v)).collect()
}

impl HmmData {
    pub fn from_json_str(s: &str) -> Result<HmmData, String> {
        let raw: RawHmm = serde_json::from_str(s).map_err(|e| e.to_string())?;

        let finalseg = FinalsegModel {
            start: char_map(raw.finalseg.start),
            trans: raw
                .finalseg
                .trans
                .into_iter()
                .map(|(k, v)| (one_char(&k), char_map(v)))
                .collect(),
            emit: raw
                .finalseg
                .emit
                .into_iter()
                .map(|(k, v)| (one_char(&k), char_map(v)))
                .collect(),
        };

        let raw_start = raw.posseg.start;
        let raw_trans = raw.posseg.trans;
        let raw_emit = raw.posseg.emit;
        let raw_char_state_tab = raw.posseg.char_state_tab;

        let mut states: Vec<StateTag> = raw_trans.keys().map(|k| StateTag::decode(k)).collect();
        states.sort();
        let state_ids: HashMap<StateTag, usize> = states
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, state)| (state, idx))
            .collect();
        let state_count = states.len();

        let mut start = vec![MIN_FLOAT; state_count];
        for (key, value) in raw_start {
            if let Some(&idx) = state_ids.get(&StateTag::decode(&key)) {
                start[idx] = value;
            }
        }

        let mut trans = vec![f64::NEG_INFINITY; state_count * state_count];
        let mut trans_from = vec![Vec::new(); state_count];
        for (from_key, targets) in raw_trans {
            let from = state_ids[&StateTag::decode(&from_key)];
            for (to_key, value) in targets {
                let to = state_ids[&StateTag::decode(&to_key)];
                trans[from * state_count + to] = value;
                trans_from[from].push(to);
            }
            trans_from[from].sort_unstable();
            trans_from[from].dedup();
        }

        let mut emit = vec![HashMap::new(); state_count];
        for (key, values) in raw_emit {
            if let Some(&idx) = state_ids.get(&StateTag::decode(&key)) {
                emit[idx] = char_map(values);
            }
        }

        let char_state_tab: HashMap<char, Vec<usize>> = raw_char_state_tab
            .into_iter()
            .map(|(k, v)| {
                let mut ids: Vec<usize> =
                    v.iter().map(|s| state_ids[&StateTag::decode(s)]).collect();
                ids.sort_unstable();
                ids.dedup();
                (one_char(&k), ids)
            })
            .collect();
        let all_states: Vec<usize> = (0..state_count).collect();

        let posseg = PossegModel {
            states,
            start,
            trans,
            trans_from,
            emit,
            char_state_tab,
            all_states,
        };

        Ok(HmmData { finalseg, posseg })
    }
}
