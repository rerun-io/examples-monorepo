#import <./consts.wgsl>

// Block scan over lds, shared by scan_common.wgsl and scan_portable.wgsl (each defines cube_scan).
var<workgroup> lds: array<u32, BLOCK_SIZE>;
fn lds_index(lin: u32) -> u32 {
    return (lin % ELEMENTS_PER_THREAD) * WG + lin / ELEMENTS_PER_THREAD;
}

fn block_scan(base: u32, inclusive: bool, lid: u32, lane: u32, width: u32) -> u32 {
    var sum = 0u;
    for (var j = 0u; j < ELEMENTS_PER_THREAD; j++) {
        let idx = j * WG + lid;
        let v = lds[idx];
        lds[idx] = sum + select(0u, v, inclusive);
        sum += v;
    }
    let offsets = cube_scan(sum, lid, lane, width);
    for (var j = 0u; j < ELEMENTS_PER_THREAD; j++) {
        lds[j * WG + lid] += base + offsets.x;
    }
    return offsets.y;
}
