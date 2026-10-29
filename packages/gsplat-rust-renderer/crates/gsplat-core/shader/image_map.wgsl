#import <./image_hexel.wgsl>
#import <./rect_map.wgsl>

@group(0) @binding(5) var<storage, read> projected: array<ImageHexel>;
@compute @workgroup_size(WG) fn image_map_tiles(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = (gid.x + gid.y * groups.x) * WG + lid;
    if i < counts[0] {
        let h = projected[i];
        map_rect_tiles(i, image_tiles(h.x, h.y, h.su, h.sv, h.sharpness));
    }
}
