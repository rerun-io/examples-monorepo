// GPU counts become portable two-dimensional dispatches. No host count is needed.
@group(0) @binding(0) var<storage, read> counts: array<u32>;
// count index, capacity, divisor, multiplier. Index MAX means a raster dispatch
// of `divisor` tiles, suppressed when the intersection count exceeds capacity.
@group(0) @binding(1) var<storage, read> plans: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> args: array<vec4u>;
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&plans) {
        return;
    }
    let plan = plans[id.x];
    var groups = 0u;
    if (counts[0] & 0x80000000u) != 0u {
        groups = 0u;
    } else if plan.x == 0xffffffffu {
        if counts[1] <= plan.y {
            groups = plan.z;
        }
    } else {
        let n = min(counts[plan.x], plan.y);
        groups = (n / plan.z + select(0u, 1u, n % plan.z != 0u)) * plan.w;
    }
    let x = max(1u, min(groups, 65535u));
    args[id.x] = vec4u(x, (groups + x - 1u) / x, 1u, 0u);
}
