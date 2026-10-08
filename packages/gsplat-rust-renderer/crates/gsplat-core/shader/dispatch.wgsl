// GPU counts become portable two-dimensional dispatches. No host count is needed.
@group(0) @binding(0) var<storage, read> counts: array<u32>;
struct DispatchPlan { count: u32, max: u32, per_group: u32, multiplier: u32, }
@group(0) @binding(1) var<storage, read> plans: array<DispatchPlan>;
@group(0) @binding(2) var<storage, read_write> args: array<vec4u>;
@compute @workgroup_size(64) fn prepare(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&plans) {
        return;
    }
    let plan = plans[id.x];
    var groups = 0u;
    if (counts[0] & 0x80000000u) != 0u {
        groups = 0u;
    } else {
        let n = min(counts[plan.count], plan.max);
        groups = (n / plan.per_group + select(0u, 1u, n % plan.per_group != 0u)) * plan.multiplier;
    }
    let x = max(1u, min(groups, 65535u));
    args[id.x] = vec4u(x, (groups + x - 1u) / x, 1u, 0u);
}
