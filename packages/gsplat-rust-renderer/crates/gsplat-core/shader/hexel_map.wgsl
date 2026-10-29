#import <./hexel_common.wgsl>
#import <./rect_map.wgsl>

@group(0) @binding(5) var<storage, read> projected: array<Hexel>;
@compute @workgroup_size(WG) fn hexel_map_tiles(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = (gid.x + gid.y * groups.x) * WG + lid;
    if i < counts[0] {
        let h = projected[i];
        map_rect_tiles(i, hexel_tiles(plane(h), h.opacity, h.sharpness));
    }
}
