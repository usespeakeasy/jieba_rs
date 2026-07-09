//! Unit tests for the parity-critical algorithms (PLAN.md T2.1–T2.7).

use crate::{Segment, Tokenizer};
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/jieba_rs/data")
}

fn tok() -> Tokenizer {
    let d = data_dir();
    Tokenizer::new(
        d.join("dict.txt"),
        d.join("hmm_data.json"),
        d.join("idf.txt"),
    )
}

fn v(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

// ---- T2.1 dictionary ----

#[test]
fn dict_totals_and_freqs() {
    let mut t = tok();
    t.initialize().unwrap();
    assert_eq!(t.total(), 60101967, "total freq");
    assert_eq!(t.get_freq("清华大学"), Some(922));
    assert_eq!(t.get_freq("清华大"), Some(0), "prefix marker");
    assert_eq!(t.get_freq("不存在的词xyz"), None);
}

// ---- T2.2 DAG + route (no HMM) ----

#[test]
fn cut_no_hmm_basic() {
    let mut t = tok();
    assert_eq!(
        t.cut("我来到北京清华大学", false, false),
        v(&["我", "来到", "北京", "清华大学"])
    );
}

// ---- T2.3 HMM finalseg ----

#[test]
fn hmm_oov() {
    let mut t = tok();
    // Full precise+HMM path; OOV names segmented by HMM.
    assert_eq!(
        t.cut("小明硕士毕业于中国科学院计算所", false, true),
        v(&["小明", "硕士", "毕业", "于", "中国科学院", "计算所"])
    );
    assert_eq!(t.cut("韩玉赏鉴", false, true), v(&["韩玉", "赏鉴"]));
}

// ---- T2.4 cut / for_search / tokenize ----

#[test]
fn cut_readme_examples() {
    let mut t = tok();
    // full mode
    assert_eq!(
        t.cut("我来到北京清华大学", true, false),
        v(&["我", "来到", "北京", "清华", "清华大学", "华大", "大学"])
    );
    // precise + HMM
    assert_eq!(
        t.cut("我来到北京清华大学", false, true),
        v(&["我", "来到", "北京", "清华大学"])
    );
}

#[test]
fn cut_for_search_example() {
    let mut t = tok();
    assert_eq!(
        t.cut_for_search("小明硕士毕业于中国科学院计算所", true),
        v(&[
            "小明",
            "硕士",
            "毕业",
            "于",
            "中国",
            "科学",
            "学院",
            "科学院",
            "中国科学院",
            "计算",
            "计算所"
        ])
    );
}

#[test]
fn tokenize_spans() {
    let mut t = tok();
    let toks = t.tokenize("永和服装饰品有限公司", false, true);
    let spans: Vec<(String, usize, usize)> = toks;
    assert_eq!(
        spans,
        vec![
            ("永和".to_string(), 0, 2),
            ("服装".to_string(), 2, 4),
            ("饰品".to_string(), 4, 6),
            ("有限公司".to_string(), 6, 10),
        ]
    );
}

// ---- T2.5 mutation ----

#[test]
fn suggest_freq_split() {
    let mut t = tok();
    t.suggest_freq(
        &Segment::Parts(vec!["中".to_string(), "将".to_string()]),
        true,
    );
    let out = t.cut("如果放到post中将出错。", false, false);
    assert!(out.contains(&"中".to_string()), "got {out:?}");
    assert!(out.contains(&"将".to_string()), "got {out:?}");
}

#[test]
fn add_word_total_inflates() {
    let mut t = tok();
    let before = t.total();
    t.add_word("石墨烯", None, None);
    // add_word adds freq to total without subtracting an old value
    assert!(t.total() >= before);
    assert!(t.get_freq("石墨烯").is_some());
    // del_word => freq 0
    t.del_word("石墨烯");
    assert_eq!(t.get_freq("石墨烯"), Some(0));
}

// ---- T2.6 posseg ----

#[test]
fn posseg_basic() {
    let mut t = tok();
    let got = t.pos_cut("我爱北京天安门", true);
    let expect: Vec<(String, String)> =
        vec![("我", "r"), ("爱", "v"), ("北京", "ns"), ("天安门", "ns")]
            .into_iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
    assert_eq!(got, expect);
}

#[test]
fn base_pos_tag_survives_frequency_mutations() {
    let mut t = tok();
    t.initialize().unwrap();

    t.set_freq("北京", 34_488);
    assert_eq!(t.pos_cut("北京", false), v2(&[("北京", "ns")]));

    // Direct FREQ deletion does not delete Jieba's separately loaded base tag.
    t.delete_freq("北京");
    t.set_freq("北京", 34_488);
    assert_eq!(t.pos_cut("北京", false), v2(&[("北京", "ns")]));

    // Re-adding without an explicit user tag also retains the base tag.
    t.add_word("北京", Some(34_488), None);
    assert_eq!(t.pos_cut("北京", false), v2(&[("北京", "ns")]));
}

#[test]
fn frequency_mutations_preserve_full_i64_range() {
    let mut t = tok();
    for freq in [i64::MIN, -(1_i64 << 55), (1_i64 << 55) - 1, i64::MAX] {
        t.set_freq("极值频率", freq);
        assert_eq!(t.get_freq("极值频率"), Some(freq));
    }
    t.delete_freq("极值频率");
    assert_eq!(t.get_freq("极值频率"), None);
    t.set_freq("极值频率", i64::MAX);
    assert_eq!(t.get_freq("极值频率"), Some(i64::MAX));
}

fn v2(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(word, tag)| ((*word).to_string(), (*tag).to_string()))
        .collect()
}

// ---- T2.7 analyse ----

#[test]
fn analyse_smoke() {
    let mut t = tok();
    let text = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5.03亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。";
    let tags = t.extract_tags(text, Some(5), &[]);
    assert!(!tags.is_empty());
    // weights strictly non-increasing
    for w in tags.windows(2) {
        assert!(w[0].1 >= w[1].1, "weights not sorted desc: {tags:?}");
    }
    let tr = t.textrank(
        text,
        Some(5),
        &["ns".into(), "n".into(), "vn".into(), "v".into()],
    );
    assert!(!tr.is_empty());
}
