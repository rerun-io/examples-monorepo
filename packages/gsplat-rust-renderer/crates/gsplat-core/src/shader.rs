//! Local equivalent of `re_renderer`'s import-once `FileResolver`; shader bytes need no rewriting.
use std::collections::HashSet;

/// The subgroup variant (tests).
#[cfg(test)]
pub(crate) fn resolve(name: &'static str) -> String {
    resolve_variant(name, true)
}

/// `name` with its imports. With `subgroups` the text is the kernels as written; without, the scan
/// helpers come from `scan_portable.wgsl` (workgroup memory only) and the module names no subgroup
/// built-in, so it needs no WGSL extension (checked by `portable_modules_need_no_subgroup_extension`).
pub(crate) fn resolve_variant(name: &'static str, subgroups: bool) -> String {
    fn append(name: &str, subgroups: bool, seen: &mut HashSet<String>, out: &mut String) {
        let name = match name {
            "scan_common.wgsl" if !subgroups => "scan_portable.wgsl",
            name => name,
        };
        if !seen.insert(name.to_owned()) {
            return;
        }
        let source = match name {
            "consts.wgsl" => include_str!("../shader/consts.wgsl"),
            "common.wgsl" => include_str!("../shader/common.wgsl"),
            "counts.wgsl" => include_str!("../shader/counts.wgsl"),
            "dispatch.wgsl" => include_str!("../shader/dispatch.wgsl"),
            "hexagon.wgsl" => include_str!("../shader/hexagon.wgsl"),
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
            "scan_block.wgsl" => include_str!("../shader/scan_block.wgsl"),
            "scan_common.wgsl" => include_str!("../shader/scan_common.wgsl"),
            "scan_portable.wgsl" => include_str!("../shader/scan_portable.wgsl"),
            "sog_dequant.wgsl" => include_str!("../shader/sog_dequant.wgsl"),
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
                append(import, subgroups, seen, out);
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    let mut out = String::new();
    append(name, subgroups, &mut HashSet::new(), &mut out);
    if subgroups {
        return out;
    }
    // The portable variant has no lane built-ins: scan_portable.wgsl defines `lane` and `width` as
    // constants, so the entry points drop those parameters, and the scatter kernel's lane-wide bin
    // prefix (dead code with `width` 1) loses its subgroup call.
    out.replace(
        ",\n    @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32)",
        ")",
    )
    .replace("subgroupInclusiveAdd(v)", "v")
}
