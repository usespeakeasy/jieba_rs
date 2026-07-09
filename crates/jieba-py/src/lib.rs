use std::sync::Mutex;

use jieba_core::{Segment, Tokenizer};
use pyo3::prelude::*;

/// Thin PyO3 wrapper around `jieba_core::Tokenizer`. Owns its dictionary, so
/// distinct instances are independent. The GIL is released around segmentation.
#[pyclass]
struct RustTokenizer {
    inner: Mutex<Tokenizer>,
}

#[pymethods]
impl RustTokenizer {
    #[new]
    fn new(dict_path: String, hmm_path: String, idf_path: String) -> Self {
        RustTokenizer {
            inner: Mutex::new(Tokenizer::new(dict_path, hmm_path, idf_path)),
        }
    }

    fn initialize(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .unwrap()
                .initialize()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
        })
    }

    fn is_initialized(&self) -> bool {
        self.inner.lock().unwrap().is_initialized()
    }

    fn set_dictionary(&self, path: String) {
        self.inner.lock().unwrap().set_dictionary(path);
    }

    fn cut(&self, py: Python<'_>, sentence: String, cut_all: bool, hmm: bool) -> Vec<String> {
        py.allow_threads(|| self.inner.lock().unwrap().cut(&sentence, cut_all, hmm))
    }

    fn cut_for_search(&self, py: Python<'_>, sentence: String, hmm: bool) -> Vec<String> {
        py.allow_threads(|| self.inner.lock().unwrap().cut_for_search(&sentence, hmm))
    }

    fn tokenize(
        &self,
        py: Python<'_>,
        sentence: String,
        search: bool,
        hmm: bool,
    ) -> Vec<(String, usize, usize)> {
        py.allow_threads(|| self.inner.lock().unwrap().tokenize(&sentence, search, hmm))
    }

    fn pos_cut(&self, py: Python<'_>, sentence: String, hmm: bool) -> Vec<(String, String)> {
        py.allow_threads(|| self.inner.lock().unwrap().pos_cut(&sentence, hmm))
    }

    fn pos_cut_detailed(
        &self,
        py: Python<'_>,
        sentence: String,
        hmm: bool,
    ) -> Vec<(String, String, bool)> {
        py.allow_threads(|| self.inner.lock().unwrap().pos_cut_detailed(&sentence, hmm))
    }

    #[pyo3(signature = (word, freq=None, tag=None))]
    fn add_word(&self, word: String, freq: Option<i64>, tag: Option<String>) {
        self.inner
            .lock()
            .unwrap()
            .add_word(&word, freq, tag.as_deref());
    }

    fn del_word(&self, word: String) {
        self.inner.lock().unwrap().del_word(&word);
    }

    fn suggest_freq_word(&self, word: String, tune: bool) -> i64 {
        self.inner
            .lock()
            .unwrap()
            .suggest_freq(&Segment::Word(word), tune)
    }

    fn suggest_freq_parts(&self, parts: Vec<String>, tune: bool) -> i64 {
        self.inner
            .lock()
            .unwrap()
            .suggest_freq(&Segment::Parts(parts), tune)
    }

    fn load_userdict_str(&self, content: String) {
        self.inner.lock().unwrap().load_userdict_str(&content);
    }

    fn set_idf_path(&self, path: String) {
        self.inner.lock().unwrap().set_idf_path(path);
    }

    fn add_stop_words(&self, words: Vec<String>) {
        self.inner.lock().unwrap().add_stop_words(words);
    }

    fn get_freq(&self, word: String) -> Option<i64> {
        self.inner.lock().unwrap().get_freq(&word)
    }

    fn set_freq(&self, word: String, freq: i64) {
        self.inner.lock().unwrap().set_freq(&word, freq);
    }

    fn delete_freq(&self, word: String) {
        self.inner.lock().unwrap().delete_freq(&word);
    }

    fn clear_freq(&self) {
        self.inner.lock().unwrap().clear_freq();
    }

    fn set_freq_entries(&self, entries: Vec<(String, i64)>) {
        self.inner.lock().unwrap().set_freq_entries(entries);
    }

    fn set_total(&self, total: i64) {
        self.inner.lock().unwrap().set_total(total);
    }

    fn add_force_split(&self, word: String) {
        self.inner.lock().unwrap().add_force_split(&word);
    }

    fn set_force_split_words(&self, words: Vec<String>) {
        self.inner.lock().unwrap().set_force_split_words(words);
    }

    fn get_dag(&self, sentence: String) -> Vec<Vec<usize>> {
        self.inner.lock().unwrap().get_dag(&sentence)
    }

    fn calc(&self, sentence: String, dag: Vec<Vec<usize>>) -> Vec<(f64, usize)> {
        self.inner.lock().unwrap().calc(&sentence, &dag)
    }

    #[getter]
    fn total(&self) -> i64 {
        self.inner.lock().unwrap().total()
    }

    #[pyo3(signature = (sentence, top_k=None, allow_pos=Vec::new()))]
    fn extract_tags(
        &self,
        py: Python<'_>,
        sentence: String,
        top_k: Option<usize>,
        allow_pos: Vec<String>,
    ) -> Vec<(String, f64)> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .unwrap()
                .extract_tags(&sentence, top_k, &allow_pos)
        })
    }

    #[pyo3(signature = (sentence, top_k=None, allow_pos=Vec::new()))]
    fn textrank(
        &self,
        py: Python<'_>,
        sentence: String,
        top_k: Option<usize>,
        allow_pos: Vec<String>,
    ) -> Vec<(String, f64)> {
        py.allow_threads(|| {
            self.inner
                .lock()
                .unwrap()
                .textrank(&sentence, top_k, &allow_pos)
        })
    }
}

#[pyfunction]
fn ping() -> &'static str {
    jieba_core::ping()
}

#[pymodule]
fn _rustjieba(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_class::<RustTokenizer>()?;
    Ok(())
}
