//! A parse result per payload generation, kept while drawn: entries not asked for in the previous frame go.
use std::collections::HashMap;

pub(crate) struct GenerationCache<T> {
    frame: u64,
    entries: HashMap<u64, (Result<T, String>, u64)>,
}

impl<T> Default for GenerationCache<T> {
    fn default() -> Self {
        Self {
            frame: 0,
            entries: HashMap::new(),
        }
    }
}

impl<T: Clone> GenerationCache<T> {
    /// Drop the entries the previous frame did not ask for.
    pub(crate) fn begin_frame(&mut self) {
        self.entries.retain(|_, (_, seen)| *seen + 1 >= self.frame);
        self.frame += 1;
    }

    /// The result for `generation`, parsed once.
    pub(crate) fn get_or_parse(
        &mut self,
        generation: u64,
        parse: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let frame = self.frame;
        let (value, seen) = self
            .entries
            .entry(generation)
            .or_insert_with(|| (parse(), frame));
        *seen = frame;
        value.clone()
    }
}
