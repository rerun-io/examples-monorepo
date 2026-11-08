//! Staged for kornia-rs / kornia-bow: what kornia-bow lacks for image retrieval.
//!
//! kornia-bow turns descriptors into bags of words ([`kornia_bow::BoW`]) and compares two of them
//! ([`kornia_bow::BoW::l1_similarity`]); [`database::Database`] adds DBoW2's inverted file, so a query scores
//! only the entries that share a word with it instead of comparing against every entry.

pub mod database;

pub use database::{Database, EntryId, QueryResult};
