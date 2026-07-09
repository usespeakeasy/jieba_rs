//! analyse: TF-IDF (`extract_tags`) and TextRank keyword extraction.
//! See docs/ALGORITHM_NOTES.md §8. Float summation order and stable sorting are
//! preserved to match jieba exactly.

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// jieba's hardcoded English stop words (KeywordExtractor.STOP_WORDS).
pub static STOP_WORDS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    [
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as",
        "on", "with", "can", "if", "from", "which", "you", "it", "this", "then", "at", "have",
        "all", "not", "one", "has", "or",
    ]
    .into_iter()
    .collect()
});

fn is_stop(word: &str, extra: &std::collections::HashSet<String>) -> bool {
    let lower = word.to_lowercase();
    STOP_WORDS.contains(lower.as_str()) || extra.contains(&lower)
}

/// Keep a word if it survives the length/stop-word filter (`len(strip) >= 2`).
fn keep_word(word: &str, extra: &std::collections::HashSet<String>) -> bool {
    word.trim().chars().count() >= 2 && !is_stop(word, extra)
}

/// Stable descending sort by weight, preserving first-seen order on ties, then
/// truncate to `top_k`. Mirrors Python `sorted(reverse=True)` (stable).
fn rank_and_truncate(mut items: Vec<(String, f64)>, top_k: Option<usize>) -> Vec<(String, f64)> {
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = top_k {
        items.truncate(k);
    }
    items
}

/// TF-IDF. `words` is (word, Option<flag>); flag is Some only when allow_pos is
/// set (posseg was used). `withFlag=False` semantics (word keys are strings).
pub fn extract_tags(
    words: Vec<(String, Option<String>)>,
    allow_pos: &[String],
    idf_freq: &HashMap<String, f64>,
    median_idf: f64,
    top_k: Option<usize>,
    extra_stop: &std::collections::HashSet<String>,
) -> Vec<(String, f64)> {
    let use_pos = !allow_pos.is_empty();
    // insertion-ordered frequency map
    let mut order: Vec<String> = Vec::new();
    let mut freq: HashMap<String, f64> = HashMap::new();

    for (w, flag) in words {
        if use_pos {
            match &flag {
                Some(f) if allow_pos.iter().any(|p| p == f) => {}
                _ => continue,
            }
        }
        if !keep_word(&w, extra_stop) {
            continue;
        }
        if let Some(v) = freq.get_mut(&w) {
            *v += 1.0;
        } else {
            order.push(w.clone());
            freq.insert(w, 1.0);
        }
    }

    let total: f64 = order.iter().map(|k| freq[k]).sum();
    let mut weighted: Vec<(String, f64)> = Vec::with_capacity(order.len());
    for k in order {
        let idf = idf_freq.get(&k).copied().unwrap_or(median_idf);
        let w = freq[&k] * idf / total;
        weighted.push((k, w));
    }
    rank_and_truncate(weighted, top_k)
}

/// TextRank. `words` is the posseg output (word, flag). `span=5`, `d=0.85`,
/// 10 iterations, then jieba's normalization.
pub fn textrank(
    words: Vec<(String, String)>,
    allow_pos: &[String],
    top_k: Option<usize>,
    extra_stop: &std::collections::HashSet<String>,
) -> Vec<(String, f64)> {
    const SPAN: usize = 5;
    const D: f64 = 0.85;

    let pairfilter = |i: usize| -> bool {
        let (word, flag) = &words[i];
        allow_pos.iter().any(|p| p == flag) && keep_word(word, extra_stop)
    };

    // co-occurrence counts, insertion-ordered
    let mut cm_order: Vec<(String, String)> = Vec::new();
    let mut cm: HashMap<(String, String), i64> = HashMap::new();
    let n = words.len();
    for i in 0..n {
        if !pairfilter(i) {
            continue;
        }
        for j in (i + 1)..(i + SPAN) {
            if j >= n {
                break;
            }
            if !pairfilter(j) {
                continue;
            }
            let key = (words[i].0.clone(), words[j].0.clone());
            if let Some(v) = cm.get_mut(&key) {
                *v += 1;
            } else {
                cm_order.push(key.clone());
                cm.insert(key, 1);
            }
        }
    }

    // build undirected graph (adjacency lists in edge-insertion order)
    let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    let mut node_order: Vec<String> = Vec::new();
    let add_node =
        |g: &mut HashMap<String, Vec<(String, f64)>>, ord: &mut Vec<String>, name: &str| {
            if !g.contains_key(name) {
                g.insert(name.to_string(), Vec::new());
                ord.push(name.to_string());
            }
        };
    for (start, end) in &cm_order {
        let w = cm[&(start.clone(), end.clone())] as f64;
        add_node(&mut graph, &mut node_order, start);
        add_node(&mut graph, &mut node_order, end);
        graph.get_mut(start).unwrap().push((end.clone(), w));
        graph.get_mut(end).unwrap().push((start.clone(), w));
    }

    if node_order.is_empty() {
        return Vec::new();
    }

    // rank
    let wsdef = 1.0 / node_order.len() as f64;
    let mut ws: HashMap<String, f64> = HashMap::new();
    let mut out_sum: HashMap<String, f64> = HashMap::new();
    for n in &node_order {
        ws.insert(n.clone(), wsdef);
        let s: f64 = graph[n].iter().fold(0.0, |acc, e| acc + e.1);
        out_sum.insert(n.clone(), s);
    }
    let mut sorted_keys = node_order.clone();
    sorted_keys.sort();

    for _ in 0..10 {
        for node in &sorted_keys {
            let mut s = 0.0f64;
            for (neigh, w) in &graph[node] {
                s += w / out_sum[neigh] * ws[neigh];
            }
            ws.insert(node.clone(), (1.0 - D) + D * s);
        }
    }

    // normalize (jieba's inverted init constants)
    let mut min_rank = f64::MAX;
    let mut max_rank = f64::MIN_POSITIVE;
    for &w in ws.values() {
        if w < min_rank {
            min_rank = w;
        }
        if w > max_rank {
            max_rank = w;
        }
    }
    let mut result: Vec<(String, f64)> = Vec::with_capacity(node_order.len());
    for node in &node_order {
        let w = ws[node];
        let norm = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0);
        result.push((node.clone(), norm));
    }
    rank_and_truncate(result, top_k)
}
