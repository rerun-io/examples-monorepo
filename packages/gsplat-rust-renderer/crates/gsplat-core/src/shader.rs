//! Local equivalent of `re_renderer`'s import-once `FileResolver`; shader bytes need no rewriting.
use std::collections::HashSet;

pub(crate) fn resolve(name: &'static str) -> String {
    fn append(name: &str, seen: &mut HashSet<String>, out: &mut String) {
        if !seen.insert(name.to_owned()) {
            return;
        }
        let source = match name {
            "consts.wgsl" => include_str!("../shader/consts.wgsl"),
            "common.wgsl" => include_str!("../shader/common.wgsl"),
            "counts.wgsl" => include_str!("../shader/counts.wgsl"),
            "dispatch.wgsl" => include_str!("../shader/dispatch.wgsl"),
            "hexel_common.wgsl" => include_str!("../shader/hexel_common.wgsl"),
            "image_hexel.wgsl" => include_str!("../shader/image_hexel.wgsl"),
            "image_map.wgsl" => include_str!("../shader/image_map.wgsl"),
            "image_project.wgsl" => include_str!("../shader/image_project.wgsl"),
            "image_raster.wgsl" => include_str!("../shader/image_raster.wgsl"),
            "hexel_map.wgsl" => include_str!("../shader/hexel_map.wgsl"),
            "hexel_project.wgsl" => include_str!("../shader/hexel_project.wgsl"),
            "hexel_raster.wgsl" => include_str!("../shader/hexel_raster.wgsl"),
            "lens.wgsl" => include_str!("../shader/lens.wgsl"),
            "map.wgsl" => include_str!("../shader/map.wgsl"),
            "project.wgsl" => include_str!("../shader/project.wgsl"),
            "raster.wgsl" => include_str!("../shader/raster.wgsl"),
            "scan.wgsl" => include_str!("../shader/scan.wgsl"),
            "scan_common.wgsl" => include_str!("../shader/scan_common.wgsl"),
            "sort.wgsl" => include_str!("../shader/sort.wgsl"),
            "raster_common.wgsl" => include_str!("../shader/raster_common.wgsl"),
            "raster_outputs.wgsl" => include_str!("../shader/raster_outputs.wgsl"),
            "rect_map.wgsl" => include_str!("../shader/rect_map.wgsl"),
            _ => panic!("unknown embedded shader import: {name}"),
        };
        for line in source.lines() {
            if let Some(import) = line
                .strip_prefix("#import <./")
                .and_then(|s| s.strip_suffix('>'))
            {
                append(import, seen, out);
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    let mut out = String::new();
    append(name, &mut HashSet::new(), &mut out);
    out
}
