//! Compiled regexes and a `re.split`-with-capturing-group helper, mirroring
//! jieba's block-splitting exactly. See docs/ALGORITHM_NOTES.md §3, §4, §7.

use once_cell::sync::Lazy;
use regex::Regex;

// Main tokenizer (jieba/__init__.py)
pub static RE_HAN_DEFAULT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\u{4E00}-\u{9FD5}a-zA-Z0-9+#&._%\-]+").unwrap());
pub static RE_SKIP_DEFAULT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\r\n|\s").unwrap());

// finalseg
pub static RE_HAN_FINALSEG: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\u{4E00}-\u{9FD5}]+").unwrap());
pub static RE_SKIP_FINALSEG: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[a-zA-Z0-9]+(?:\.\d+)?%?").unwrap());

// posseg
pub static RE_HAN_INTERNAL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\u{4E00}-\u{9FD5}a-zA-Z0-9+#&._]+").unwrap());
pub static RE_SKIP_INTERNAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\r\n|\s").unwrap());
pub static RE_HAN_DETAIL: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\u{4E00}-\u{9FD5}]+").unwrap());
pub static RE_SKIP_DETAIL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\.0-9]+|[a-zA-Z0-9]+").unwrap());

/// One piece of a `re.split` result.
pub struct Seg<'a> {
    pub text: &'a str,
    /// True if this piece was a regex match (a captured group in Python).
    pub is_match: bool,
}

/// Emulate Python `re.split(r"(pattern)", text)` where the whole pattern is one
/// capturing group: returns the between-text and the matched text interleaved,
/// including the empty strings Python inserts at boundaries.
pub fn split_captured<'a>(re: &Regex, text: &'a str) -> Vec<Seg<'a>> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push(Seg {
            text: &text[last..m.start()],
            is_match: false,
        });
        out.push(Seg {
            text: &text[m.start()..m.end()],
            is_match: true,
        });
        last = m.end();
    }
    out.push(Seg {
        text: &text[last..],
        is_match: false,
    });
    out
}

/// `re.match(pattern, s)` — anchored at start — for the han patterns: true iff
/// `s` begins with at least one character the pattern accepts. Since all han
/// patterns are `[...]+`, a start-anchored find is equivalent to testing the
/// first char; but we test via the regex to stay faithful to the class.
pub fn matches_at_start(re: &Regex, s: &str) -> bool {
    match re.find(s) {
        Some(m) => m.start() == 0,
        None => false,
    }
}

/// `re_eng.match` for `[a-zA-Z0-9]` (and posseg `[a-zA-Z0-9]+`): starts with an
/// ASCII alphanumeric.
#[inline]
pub fn starts_alnum(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
}

/// `re_num.match` for `[\.0-9]+`: starts with '.' or an ASCII digit.
#[inline]
pub fn starts_num(s: &str) -> bool {
    s.chars()
        .next()
        .is_some_and(|c| c == '.' || c.is_ascii_digit())
}

/// `re_eng1` = `^[a-zA-Z0-9]$`: exactly one ASCII alphanumeric char.
#[inline]
pub fn is_single_alnum(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) => c.is_ascii_alphanumeric() && it.next().is_none(),
        None => false,
    }
}
