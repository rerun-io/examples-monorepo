// Hand port of brush-scan 1388f74c; adapted to raw WGSL.
var<workgroup> partials: array<u32, 64>;
var<workgroup> cube_total: u32;
var<workgroup> lds: array<u32, 1024>;
fn cube_scan(value: u32, lid: u32, lane: u32, width: u32) -> vec2u {
    let plane = lid / width;
    let planes = 256u / width;
    let inclusive = subgroupInclusiveAdd(value);
    if lane == width - 1u {
        partials[plane] = inclusive;
    }
    workgroupBarrier();
    if planes <= width {
        var v = 0u;
        if lane < planes {
            v = partials[lane];
        }
        let scanned = subgroupExclusiveAdd(v);
        // Every plane reads before plane zero overwrites the shared totals.
        workgroupBarrier();
        if plane == 0u {
            if lane < planes {
                partials[lane] = scanned;
            }
            if lane == planes - 1u {
                cube_total = scanned + v;
            }
        }
    } else {
        if lid == 0u {
            var acc = 0u;
            for (var i = 0u; i < planes; i++) {
                let v = partials[i];
                partials[i] = acc;
                acc += v;
            }
            cube_total = acc;
        }
    }
    workgroupBarrier();
    return vec2u(partials[plane] + inclusive - value, cube_total);
}

fn lds_index(lin: u32) -> u32 {
    return (lin % 4u) * 256u + lin / 4u;
}

fn block_scan(base: u32, inclusive: bool, lid: u32, lane: u32, width: u32) -> u32 {
    var sum = 0u;
    for (var j = 0u; j < 4u; j++) {
        let idx = j * 256u + lid;
        let v = lds[idx];
        lds[idx] = sum + select(0u, v, inclusive);
        sum += v;
    }
    let offsets = cube_scan(sum, lid, lane, width);
    for (var j = 0u; j < 4u; j++) {
        lds[j * 256u + lid] += base + offsets.x;
    }
    return offsets.y;
}
