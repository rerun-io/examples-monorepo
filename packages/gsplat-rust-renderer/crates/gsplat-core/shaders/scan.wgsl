// Hand port of brush-scan 1388f74c.
struct Params {
    count_index: u32,
    divisor: u32,
    capacity: u32,
    pad: u32
}

@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> length: u32;
@compute @workgroup_size(256) fn scan(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) width: u32) {
    if lid == 0u {
        length =(min(count[params.count_index], params.capacity) + params.divisor - 1u) / params.divisor;
    }
    let n = workgroupUniformLoad(& length);
    let block = gid.x + gid.y * groups.x;
    let base = block * 1024u;
    if base >= n {
        return;
    }
    for (var j = 0u; j < 4u; j ++) {
        let lin = j * 256u + lid;
        var v = 0u;
        if base + lin < n {
            v = input[base + lin];
        }
        lds[lds_index(lin)] = v;
    }
    workgroupBarrier();
    let total = block_scan(0u, true, lid, lane, width);
    workgroupBarrier();
    for (var j = 0u; j < 4u; j ++) {
        let lin = j * 256u + lid;
        if base + lin < n {
            output[base + lin] = lds[lds_index(lin)];
        }
    }
    if lid == 0u {
        sums[block] = total;
    }
}

@compute @workgroup_size(256) fn add_offsets(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let idx =(gid.x + gid.y * groups.x) * 256u + lid;
    let n =(min(count[params.count_index], params.capacity) + params.divisor - 1u) / params.divisor;
    if idx < n && idx >= 1024u {
        output[idx] += input[idx / 1024u - 1u];
    }
}
