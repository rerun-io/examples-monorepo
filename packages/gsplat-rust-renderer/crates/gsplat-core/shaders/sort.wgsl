// Hand port of brush-sort 1388f74c. Five stages per 4-bit digit.
struct Params { count_index: u32, shift: u32, capacity: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> reduced: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;
var<workgroup> length: u32;
var<workgroup> histogram: array<atomic<u32>, 16>;
var<workgroup> local_keys: array<u32, 256>;
var<workgroup> local_values: array<u32, 256>;
var<workgroup> bin_offsets: array<u32, 16>;
var<workgroup> bin_prefix: array<u32, 16>;
fn sort_length(lid: u32) -> u32 {
    if lid == 0u { length = min(count[params.count_index], params.capacity); }
    return workgroupUniformLoad(&length);
}
@compute @workgroup_size(256)
fn count_keys(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n = sort_length(lid);
    let nw = (n + 1023u) / 1024u;
    let group = gid.x + gid.y * groups.x;
    if group >= nw { return; }
    if lid < 16u { atomicStore(&histogram[lid], 0u); }
    workgroupBarrier();
    for (var j = 0u; j < 4u; j++) {
        let i = group * 1024u + j * 256u + lid;
        if i < n { atomicAdd(&histogram[(src[i] >> params.shift) & 15u], 1u); }
    }
    workgroupBarrier();
    if lid < 16u { counts[lid * nw + group] = atomicLoad(&histogram[lid]); }
}
@compute @workgroup_size(256)
fn reduce_counts(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u,
 @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
    let n = sort_length(lid);
    let nw = (n + 1023u) / 1024u;
    let per_bin = (nw + 1023u) / 1024u;
    let group = gid.x + gid.y * groups.x;
    if group >= 16u * per_bin { return; }
    let bin = group / per_bin;
    let base = (group % per_bin) * 1024u;
    var sum = 0u;
    for (var j = 0u; j < 4u; j++) {
        let i = base + j * 256u + lid;
        if i < nw { sum += counts[bin * nw + i]; }
    }
    let result = cube_scan(sum, lid, lane, width);
    if lid == 0u { reduced[group] = result.y; }
}
@compute @workgroup_size(256)
fn scan_counts(@builtin(local_invocation_index) lid: u32,
 @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
    let n = sort_length(lid);
    let nr = 16u * (((n + 1023u) / 1024u + 1023u) / 1024u);
    var carry = 0u;
    for (var base = 0u; base < nr; base += 1024u) {
        for (var j = 0u; j < 4u; j++) {
            let lin = j * 256u + lid;
            var v = 0u;
            if base + lin < nr { v = reduced[base + lin]; }
            lds[lds_index(lin)] = v;
        }
        workgroupBarrier();
        let total = block_scan(carry, false, lid, lane, width);
        workgroupBarrier();
        for (var j = 0u; j < 4u; j++) {
            let lin = j * 256u + lid;
            if base + lin < nr { reduced[base + lin] = lds[lds_index(lin)]; }
        }
        workgroupBarrier();
        carry += total;
    }
}
@compute @workgroup_size(256)
fn scan_add(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u,
 @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
    let n = sort_length(lid);
    let nw = (n + 1023u) / 1024u;
    let per_bin = (nw + 1023u) / 1024u;
    let group = gid.x + gid.y * groups.x;
    if group >= 16u * per_bin { return; }
    let bin = group / per_bin;
    let base = (group % per_bin) * 1024u;
    for (var j = 0u; j < 4u; j++) {
        let lin = j * 256u + lid;
        var v = 0u;
        if base + lin < nw { v = counts[bin * nw + base + lin]; }
        lds[lds_index(lin)] = v;
    }
    workgroupBarrier();
    block_scan(reduced[group], false, lid, lane, width);
    workgroupBarrier();
    for (var j = 0u; j < 4u; j++) {
        let lin = j * 256u + lid;
        if base + lin < nw { counts[bin * nw + base + lin] = lds[lds_index(lin)]; }
    }
}
@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u,
 @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
    let n = sort_length(lid);
    let nw = (n + 1023u) / 1024u;
    let group = gid.x + gid.y * groups.x;
    if group >= nw { return; }
    if lid < 16u { bin_offsets[lid] = counts[lid * nw + group]; }
    workgroupBarrier();
    for (var j = 0u; j < 4u; j++) {
        if lid < 16u { atomicStore(&histogram[lid], 0u); }
        let i = group * 1024u + j * 256u + lid;
        var key = 0xffffffffu;
        var value = 0u;
        if i < n { key = src[i]; value = values[i]; }
        for (var bit = 0u; bit < 4u; bit += 2u) {
            let pair = (key >> (params.shift + bit)) & 3u;
            let result = cube_scan(1u << (pair * 8u), lid, lane, width);
            let offsets = (result.y << 8u) + (result.y << 16u) + (result.y << 24u);
            let pos = ((offsets + result.x) >> (pair * 8u)) & 255u;
            local_keys[pos] = key;
            local_values[pos] = value;
            workgroupBarrier();
            key = local_keys[lid]; value = local_values[lid];
        }
        let bin = (key >> params.shift) & 15u;
        atomicAdd(&histogram[bin], 1u);
        workgroupBarrier();
        if width >= 16u {
            var v = 0u;
            if lane < 16u { v = atomicLoad(&histogram[lane]); }
            let prefix = subgroupInclusiveAdd(v);
            if lid < 16u { bin_prefix[lid] = prefix; }
        } else {
            if lid == 0u {
                var acc = 0u;
                for (var b = 0u; b < 16u; b++) { acc += atomicLoad(&histogram[b]); bin_prefix[b] = acc; }
            }
        }
        workgroupBarrier();
        let global_offset = bin_offsets[bin];
        workgroupBarrier();
        var local_offset = lid;
        if bin > 0u { local_offset -= bin_prefix[bin - 1u]; }
        let pos = global_offset + local_offset;
        if pos < n { dst[pos] = key; out_values[pos] = value; }
        if lid < 16u { bin_offsets[lid] += atomicLoad(&histogram[lid]); }
        workgroupBarrier();
    }
}
