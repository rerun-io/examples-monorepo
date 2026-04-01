// GPU radix-sort over tile intersection ids.
//
// Input: unsorted per-intersection tile ids emitted by the map stage.
// Output: tile ids grouped contiguously so the later tile-offset and raster passes can process
// one tile at a time.

struct SortUniformBuffer {
    shift: u32,
    total_keys_unused: u32,
    pad: vec2u,
};

struct ScanUniformBuffer {
    total_selected: u32,
    block_count: u32,
    pad: vec2u,
};

@group(0) @binding(0) var<uniform> sort_uniforms: SortUniformBuffer;
@group(0) @binding(6) var<storage, read> sort_total_keys: array<u32>;

const WG: u32 = 256u;
const BITS_PER_PASS: u32 = 4u;
const BIN_COUNT: u32 = 1u << BITS_PER_PASS;
const ELEMENTS_PER_THREAD: u32 = 1u;
const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;

fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

fn get_workgroup_id(wid: vec3u, num_wgs: vec3u) -> u32 {
    return wid.x + wid.y * num_wgs.x;
}

fn num_keys() -> u32 {
    return sort_total_keys[0];
}

@group(0) @binding(1) var<storage, read> sort_src_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> sort_counts: array<u32>;

var<workgroup> histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG, 1, 1)
fn sort_count_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = num_keys();
    let num_wgs = div_ceil(total, BLOCK_SIZE);
    let group_id = get_workgroup_id(wid, num_workgroups);

    if group_id >= num_wgs {
        return;
    }

    if local_id.x < BIN_COUNT {
        histogram[local_id.x] = 0u;
    }
    workgroupBarrier();

    let wg_block_start = BLOCK_SIZE * group_id;
    var data_index = wg_block_start + local_id.x;

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        if data_index < total {
            let local_key = (sort_src_keys[data_index] >> sort_uniforms.shift) & 0xfu;
            atomicAdd(&histogram[local_key], 1u);
        }
        data_index += WG;
    }

    workgroupBarrier();
    if local_id.x < BIN_COUNT {
        sort_counts[local_id.x * num_wgs + group_id] = histogram[local_id.x];
    }
}

@group(0) @binding(1) var<storage, read> reduce_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduce_out: array<u32>;

var<workgroup> reduce_sums: array<u32, WG>;

@compute
@workgroup_size(WG, 1, 1)
fn sort_reduce_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = num_keys();
    let num_wgs = div_ceil(total, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);
    let group_id = get_workgroup_id(wid, num_workgroups);

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = max(num_reduce_wgs / BIN_COUNT, 1u);
    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * BLOCK_SIZE;
    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        if data_index < num_wgs {
            sum += reduce_counts[bin_offset + data_index];
        }
    }
    reduce_sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x < ((WG / 2u) >> i) {
            sum += reduce_sums[local_id.x + ((WG / 2u) >> i)];
            reduce_sums[local_id.x] = sum;
        }
    }
    if local_id.x == 0u {
        reduce_out[group_id] = sum;
    }
}

@group(0) @binding(1) var<storage, read_write> scan_reduced: array<u32>;

var<workgroup> scan_sums: array<u32, WG>;
var<workgroup> scan_lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@group(0) @binding(8) var<storage, read> compose_local_offsets: array<u32>;
@group(0) @binding(9) var<storage, read> compose_block_offsets: array<u32>;
@group(0) @binding(10) var<storage, read_write> compose_out: array<u32>;
@group(0) @binding(11) var<uniform> compose_uniforms: ScanUniformBuffer;

@compute
@workgroup_size(WG, 1, 1)
fn sort_scan_main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let total = num_keys();
    let num_wgs = div_ceil(total, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        scan_lds[row][col] = select(0u, scan_reduced[data_index], data_index < num_reduce_wgs);
    }
    workgroupBarrier();

    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = scan_lds[i][local_id.x];
        scan_lds[i][local_id.x] = sum;
        sum += tmp;
    }
    scan_sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += scan_sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scan_sums[local_id.x] = sum;
    }
    workgroupBarrier();

    sum = 0u;
    if local_id.x > 0u {
        sum = scan_sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        scan_lds[i][local_id.x] += sum;
    }
    workgroupBarrier();

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_reduce_wgs {
            scan_reduced[data_index] = scan_lds[row][col];
        }
    }
}

@compute
@workgroup_size(WG, 1, 1)
fn sort_scan_compose_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = compose_uniforms.total_selected;
    let absolute_index = get_workgroup_id(wid, num_workgroups) * WG + local_id.x;
    if absolute_index >= total {
        return;
    }

    let block_index = absolute_index / (WG * 2u);
    compose_out[absolute_index] =
        compose_local_offsets[absolute_index] + compose_block_offsets[block_index];
}

@group(0) @binding(1) var<storage, read> scan_add_reduced: array<u32>;
@group(0) @binding(2) var<storage, read_write> scan_add_counts: array<u32>;

var<workgroup> scan_add_sums: array<u32, WG>;
var<workgroup> scan_add_lds: array<array<u32, WG>, ELEMENTS_PER_THREAD>;

@compute
@workgroup_size(WG, 1, 1)
fn sort_scan_add_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = num_keys();
    let num_wgs = div_ceil(total, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);
    let group_id = get_workgroup_id(wid, num_workgroups);

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = max(num_reduce_wgs / BIN_COUNT, 1u);
    let bin_id = group_id / num_reduce_wg_per_bin;
    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        scan_add_lds[row][col] = select(0u, scan_add_counts[bin_offset + data_index], data_index < num_wgs);
    }
    workgroupBarrier();

    var sum = 0u;
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let tmp = scan_add_lds[i][local_id.x];
        scan_add_lds[i][local_id.x] = sum;
        sum += tmp;
    }
    scan_add_sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += scan_add_sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scan_add_sums[local_id.x] = sum;
    }
    workgroupBarrier();

    sum = scan_add_reduced[group_id];
    if local_id.x > 0u {
        sum += scan_add_sums[local_id.x - 1u];
    }
    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        scan_add_lds[i][local_id.x] += sum;
    }
    workgroupBarrier();

    for (var i = 0u; i < ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_wgs {
            scan_add_counts[bin_offset + data_index] = scan_add_lds[row][col];
        }
    }
}

@group(0) @binding(1) var<storage, read> scatter_src_keys: array<u32>;
@group(0) @binding(2) var<storage, read> scatter_src_values: array<u32>;
@group(0) @binding(3) var<storage, read> scatter_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> scatter_out_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> scatter_out_values: array<u32>;

var<workgroup> scatter_lds_sums: array<u32, WG>;
var<workgroup> scatter_lds_scratch: array<u32, WG>;
var<workgroup> scatter_bin_offset_cache: array<u32, WG>;
var<workgroup> scatter_local_histogram: array<atomic<u32>, BIN_COUNT>;

@compute
@workgroup_size(WG, 1, 1)
fn sort_scatter_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = num_keys();
    let num_wgs = div_ceil(total, BLOCK_SIZE);
    let group_id = get_workgroup_id(wid, num_workgroups);
    if group_id >= num_wgs {
        return;
    }

    if local_id.x < BIN_COUNT {
        scatter_local_histogram[local_id.x] = 0u;
    }
    workgroupBarrier();

    let block_start = group_id * BLOCK_SIZE;
    if local_id.x < BIN_COUNT {
        scatter_bin_offset_cache[local_id.x] = scatter_counts[local_id.x * num_wgs + group_id];
    }
    workgroupBarrier();

    var key = 0u;
    var value = 0u;
    var key_digit = 0u;
    let data_index = block_start + local_id.x;
    let valid = data_index < total;
    if valid {
        key = scatter_src_keys[data_index];
        value = scatter_src_values[data_index];
        key_digit = (key >> sort_uniforms.shift) & 0xfu;
    }

    let digit_mask = select(0u, 1u, valid);
    scatter_lds_scratch[local_id.x] = digit_mask;
    workgroupBarrier();

    for (var bin = 0u; bin < BIN_COUNT; bin++) {
        let in_bin = select(0u, 1u, valid && key_digit == bin);
        scatter_lds_sums[local_id.x] = in_bin;
        workgroupBarrier();

        var offset = in_bin;
        for (var step = 1u; step < WG; step <<= 1u) {
            let add = select(0u, scatter_lds_sums[local_id.x - step], local_id.x >= step);
            workgroupBarrier();
            scatter_lds_sums[local_id.x] += add;
            workgroupBarrier();
            offset = scatter_lds_sums[local_id.x];
        }

        if valid && key_digit == bin {
            let base = scatter_bin_offset_cache[bin];
            let dst = base + offset - 1u;
            scatter_out_keys[dst] = key;
            scatter_out_values[dst] = value;
        }

        workgroupBarrier();
        if local_id.x == WG - 1u {
            scatter_bin_offset_cache[bin] += scatter_lds_sums[local_id.x];
        }
        workgroupBarrier();
    }
}
