#import <./scan_block.wgsl>

// Workgroup-memory form of scan_common.wgsl for WebGPU devices without lane built-ins (WebKit on
// iPadOS): same functions, same exact integer results. shader.rs drops the lane parameters from the
// entry points of this variant, so these constants stand in for them; with width 1 the scatter
// kernel takes its serial bin prefix.
const lane = 0u;
const width = 1u;

var<workgroup> scan_lanes: array<u32, WG>;
// Exclusive prefix and total of `value` over the workgroup (Hillis-Steele, log2(WG) steps).
fn cube_scan(value: u32, lid: u32, lane: u32, width: u32) -> vec2u {
    scan_lanes[lid] = value;
    for (var offset = 1u; offset < WG; offset <<= 1u) {
        workgroupBarrier();
        var sum = scan_lanes[lid];
        if lid >= offset {
            sum += scan_lanes[lid - offset];
        }
        workgroupBarrier();
        scan_lanes[lid] = sum;
    }
    workgroupBarrier();
    let inclusive = scan_lanes[lid];
    let total = scan_lanes[WG - 1u];
    workgroupBarrier();
    return vec2u(inclusive - value, total);
}
