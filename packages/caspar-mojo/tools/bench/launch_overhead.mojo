# Launch-overhead probe: the PCG design depends on what one launch, one sync and one scalar read-back cost.
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, ImmTileTensor, MutTileTensor, row_major

comptime BLOCK = 256
comptime L1 = type_of(row_major((0,)))


def axpy(
    a: Float32,
    x: ImmTileTensor[.float32, L1, ImmUntrackedOrigin],
    y: MutTileTensor[.float32, L1, MutUntrackedOrigin],
    n: Int32,
):
    comptime assert x.flat_rank == 1 and y.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        y[i] = a * x[i] + y[i]


def axpy_ct[a: Float32](
    x: ImmTileTensor[.float32, L1, ImmUntrackedOrigin],
    y: MutTileTensor[.float32, L1, MutUntrackedOrigin],
    n: Int32,
):
    comptime assert x.flat_rank == 1 and y.flat_rank == 1
    var i = global_idx.x
    if i < Int(n):
        y[i] = a * x[i] + y[i]


def check(ctx: DeviceContext, n: Int) raises:
    var x_buf = ctx.enqueue_create_buffer[.float32](n)
    var y_buf = ctx.enqueue_create_buffer[.float32](n)
    with x_buf.map_to_host() as xh, y_buf.map_to_host() as yh:
        for i in range(n):
            xh[i] = Float32(i)
            yh[i] = 1.0
    var x = TileTensor(x_buf, row_major(n)).as_imm()
    var y = TileTensor(y_buf, row_major(n))
    ctx.enqueue_function[axpy](Float32(2.0), x, y, Int32(n), grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
    comptime kernel_ct = axpy_ct[3.0]
    ctx.enqueue_function[kernel_ct](x, y, Int32(n), grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
    with y_buf.map_to_host() as yh:
        for i in range(n):
            var want = 5.0 * Float32(i) + 1.0
            if abs(yh[i] - want) > 1e-6 * abs(want) + 1e-6:
                raise Error(t"axpy mismatch at {i}: {yh[i]} != {want}")
    print("check: axpy and axpy_ct[3.0] correct on n =", n)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    print("device:", ctx.device().get_name(), " api:", ctx.api())
    check(ctx, 1000003)

    comptime launches = 10000
    comptime small = 1024
    var xs_buf = ctx.enqueue_create_buffer[.float32](small)
    var ys_buf = ctx.enqueue_create_buffer[.float32](small)
    xs_buf.enqueue_fill(1.0)
    ys_buf.enqueue_fill(0.0)
    var xs = TileTensor(xs_buf, row_major(small)).as_imm()
    var ys = TileTensor(ys_buf, row_major(small))
    var g = ceildiv(small, BLOCK)
    var host_one = ctx.enqueue_create_host_buffer[.float32](1)
    var dev_one = ys_buf.create_sub_buffer[.float32](0, 1)
    ctx.synchronize()

    # Best of 3 repeats per measurement: the first repeat includes module load and clock ramp-up.
    var best_a = Float64.MAX
    var best_b = Float64.MAX
    var best_c_map = Float64.MAX
    var best_c_copy = Float64.MAX
    for _ in range(3):
        var t0 = perf_counter_ns()
        for _ in range(launches):
            ctx.enqueue_function[axpy](Float32(1.0), xs, ys, Int32(small), grid_dim=g, block_dim=BLOCK)
        ctx.synchronize()
        best_a = min(best_a, Float64(perf_counter_ns() - t0) / 1e3 / Float64(launches))

        t0 = perf_counter_ns()
        for _ in range(launches):
            ctx.enqueue_function[axpy](Float32(1.0), xs, ys, Int32(small), grid_dim=g, block_dim=BLOCK)
            ctx.synchronize()
        best_b = min(best_b, Float64(perf_counter_ns() - t0) / 1e3 / Float64(launches))

        var sink: Float32 = 0.0
        t0 = perf_counter_ns()
        for _ in range(launches):
            ctx.enqueue_function[axpy](Float32(1.0), xs, ys, Int32(small), grid_dim=g, block_dim=BLOCK)
            with dev_one.map_to_host() as h:
                sink += h[0]
        best_c_map = min(best_c_map, Float64(perf_counter_ns() - t0) / 1e3 / Float64(launches))

        t0 = perf_counter_ns()
        for _ in range(launches):
            ctx.enqueue_function[axpy](Float32(1.0), xs, ys, Int32(small), grid_dim=g, block_dim=BLOCK)
            ctx.enqueue_copy(host_one, dev_one)
            ctx.synchronize()
            sink += host_one[0]
        best_c_copy = min(best_c_copy, Float64(perf_counter_ns() - t0) / 1e3 / Float64(launches))
        if sink < 0.0:
            print(sink)

    comptime big = 16 * 1024 * 1024
    var xb_buf = ctx.enqueue_create_buffer[.float32](big)
    var yb_buf = ctx.enqueue_create_buffer[.float32](big)
    xb_buf.enqueue_fill(1.0)
    yb_buf.enqueue_fill(0.0)
    var xb = TileTensor(xb_buf, row_major(big)).as_imm()
    var yb = TileTensor(yb_buf, row_major(big))
    ctx.synchronize()
    var best_d = Float64.MAX
    comptime reps = 50
    for _ in range(3):
        var t0 = perf_counter_ns()
        for _ in range(reps):
            ctx.enqueue_function[axpy](Float32(1.0), xb, yb, Int32(big), grid_dim=ceildiv(big, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
        best_d = min(best_d, Float64(perf_counter_ns() - t0) / 1e3 / Float64(reps))
    # axpy moves 3 floats per element (read x, read y, write y).
    var gbps = Float64(3 * 4 * big) / (best_d * 1e-6) / 1e9

    print("(a) launch, one sync at end      us/launch:", best_a)
    print("(b) launch + sync each           us/launch:", best_b)
    print("(c1) launch + map_to_host 1 float us/iter:", best_c_map)
    print("(c2) launch + enqueue_copy 1 float + sync us/iter:", best_c_copy)
    print("(d) axpy n=16M                    us/launch:", best_d, " GB/s:", gbps)
