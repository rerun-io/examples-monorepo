//! lod-meta.json (Streamed SOG version 1): the tree of leaf boxes and, per leaf and level, its run in a file.
use serde::Deserialize;

/// lod-meta.json failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("lod-meta.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("lod-meta.json version {0} is not supported (version 1 only)")]
    Version(u32),
    #[error("lod-meta.json: {0}")]
    Invalid(String),
}

/// Most levels a set may declare. splat-transform writes about six; each leaf holds one slot per level, so an
/// unchecked count from a file would size every leaf's allocation.
pub const MAX_LEVELS: u32 = 32;

/// A run of rows inside one file of the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct FileRun {
    pub file: u32,
    pub offset: u32,
    pub count: u32,
}

/// One tree leaf: its box and, per level (index = level), the run of that level in its file.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafMeta {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub lods: Vec<Option<FileRun>>,
}

/// lod-meta.json (Streamed SOG version 1). Unknown fields are ignored.
#[derive(Clone, Debug)]
pub struct LodMeta {
    /// All rows of all levels.
    pub count: u64,
    pub levels: u32,
    /// SOG meta.json paths relative to the lod-meta.json directory, indexed by `FileRun::file`.
    pub filenames: Vec<String>,
    /// Environment SOG (always drawn by `PlayCanvas`); not loaded here.
    pub environment: Option<String>,
    /// Leaves in depth-first tree order.
    pub leaves: Vec<LeafMeta>,
}

#[derive(Deserialize)]
struct Bound {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Deserialize)]
struct Node {
    bound: Bound,
    #[serde(default)]
    children: Vec<Self>,
    lods: Option<std::collections::BTreeMap<String, FileRun>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMeta {
    version: u32,
    count: u64,
    lod_levels: u32,
    filenames: Vec<String>,
    environment: Option<String>,
    tree: Node,
}

impl LodMeta {
    pub fn parse(json: &str) -> Result<Self, Error> {
        let raw: RawMeta = serde_json::from_str(json)?;
        if raw.version != 1 {
            return Err(Error::Version(raw.version));
        }
        if raw.lod_levels > MAX_LEVELS {
            return Err(Error::Invalid(format!(
                "{} levels, more than the {MAX_LEVELS} a set can hold",
                raw.lod_levels
            )));
        }
        let mut leaves = Vec::new();
        let mut stack = vec![&raw.tree];
        while let Some(node) = stack.pop() {
            if let Some(lods) = &node.lods {
                let mut runs = vec![None; raw.lod_levels as usize];
                for (key, run) in lods {
                    let level: usize = key
                        .parse()
                        .map_err(|e| Error::Invalid(format!("level key {key:?}: {e}")))?;
                    if level >= runs.len() || run.file as usize >= raw.filenames.len() {
                        return Err(Error::Invalid(format!(
                            "leaf run {run:?} at level {level} is outside {} levels / {} files",
                            raw.lod_levels,
                            raw.filenames.len()
                        )));
                    }
                    runs[level] = Some(*run);
                }
                leaves.push(LeafMeta {
                    min: node.bound.min,
                    max: node.bound.max,
                    lods: runs,
                });
            }
            stack.extend(node.children.iter().rev());
        }
        Ok(Self {
            count: raw.count,
            levels: raw.lod_levels,
            filenames: raw.filenames,
            environment: raw.environment,
            leaves,
        })
    }

    /// The level of each file (the level under which leaves reference it); `None` for unreferenced files.
    pub fn file_levels(&self) -> Result<Vec<Option<u32>>, Error> {
        let mut levels = vec![None; self.filenames.len()];
        for leaf in &self.leaves {
            for (level, run) in leaf.lods.iter().enumerate() {
                if let Some(run) = run {
                    let slot = &mut levels[run.file as usize];
                    match *slot {
                        Some(l) if l != level as u32 => {
                            return Err(Error::Invalid(format!(
                                "file {} holds levels {l} and {level}",
                                run.file
                            )));
                        }
                        _ => *slot = Some(level as u32),
                    }
                }
            }
        }
        Ok(levels)
    }

    /// The files a load keeps, as `(file, level)` in `filenames` order: referenced files of a level >= `min_level`.
    /// A `min_level` past the set's coarsest level is an error, not an empty scene.
    pub fn loaded_files(&self, min_level: u32) -> Result<Vec<(usize, u32)>, Error> {
        let levels = self.file_levels()?;
        let kept: Vec<(usize, u32)> = levels
            .iter()
            .enumerate()
            .filter_map(|(file, level)| level.filter(|&l| l >= min_level).map(|l| (file, l)))
            .collect();
        if kept.is_empty() && levels.iter().any(Option::is_some) {
            return Err(Error::Invalid(format!(
                "minimum level {min_level} is past the coarsest level of the set ({} levels)",
                self.levels
            )));
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_streamed_sog_tree() {
        let json = r#"{"version":1,"count":30,"lodLevels":2,"lodErrors":true,"filenames":["0_0/meta.json","1_0/meta.json"],
          "tree":{"bound":{"min":[0,0,0],"max":[2,1,1]},"children":[
            {"bound":{"min":[0,0,0],"max":[1,1,1]},"lods":{"0":{"file":0,"offset":0,"count":10},"1":{"file":1,"offset":0,"count":5}},"errors":[0,1]},
            {"bound":{"min":[1,0,0],"max":[2,1,1]},"lods":{"0":{"file":0,"offset":10,"count":10},"1":{"file":1,"offset":5,"count":5}}}]}}"#;
        let meta = LodMeta::parse(json).unwrap();
        assert_eq!(meta.levels, 2);
        assert_eq!(meta.leaves.len(), 2);
        assert_eq!(meta.leaves[1].min, [1.0, 0.0, 0.0]);
        assert_eq!(
            meta.leaves[1].lods[1],
            Some(FileRun {
                file: 1,
                offset: 5,
                count: 5
            })
        );
        assert_eq!(meta.file_levels().unwrap(), vec![Some(0), Some(1)]);
        assert_eq!(meta.loaded_files(1).unwrap(), vec![(1, 1)]);
        // A minimum level past the coarsest would load an empty scene.
        assert!(matches!(meta.loaded_files(2), Err(Error::Invalid(_))));
        assert!(LodMeta::parse(&json.replace("\"version\":1", "\"version\":2")).is_err());
    }

    #[test]
    fn refuses_more_levels_than_a_set_can_hold() {
        // Each leaf allocates one slot per level, so the count from the file is bounded before any leaf is read.
        let json = |levels: u64| {
            format!(
                r#"{{"version":1,"count":1,"lodLevels":{levels},"filenames":["0_0/meta.json"],
                "tree":{{"bound":{{"min":[0,0,0],"max":[1,1,1]}},"lods":{{"0":{{"file":0,"offset":0,"count":1}}}}}}}}"#
            )
        };
        assert_eq!(
            LodMeta::parse(&json(u64::from(MAX_LEVELS))).unwrap().levels,
            MAX_LEVELS
        );
        assert!(matches!(
            LodMeta::parse(&json(u64::from(MAX_LEVELS) + 1)),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            LodMeta::parse(&json(4_000_000_000)),
            Err(Error::Invalid(_))
        ));
    }
}
