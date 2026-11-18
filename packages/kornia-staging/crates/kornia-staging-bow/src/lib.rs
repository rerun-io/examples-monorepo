//! Staged for kornia-rs / kornia-bow: what kornia-bow lacks for image retrieval.
//!
//! kornia-bow turns descriptors into bags of words ([`kornia_bow::BoW`]) and compares two of them
//! ([`kornia_bow::BoW::l1_similarity`]); [`database::Database`] adds DBoW2's inverted file, so a query scores
//! only the entries that share a word with it instead of comparing against every entry. [`hash_bow::HashBow`] is
//! RootSLAM's vocabulary-free bag of words for 256-bit descriptors on top of that database, keyed by frame and
//! camera, with the minimum-age filter and the keyframe removal a loop closer needs.

pub mod database;
pub mod hash_bow;

pub use database::{Database, EntryId, QueryResult};
