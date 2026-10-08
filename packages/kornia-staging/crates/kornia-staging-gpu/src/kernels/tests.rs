use super::{finite, trig};
use crate::{runtime::gpu_client, GpuRuntime};
use cubecl::frontend::CompilationArg;
#[cubecl::prelude::cube(launch_unchecked)]
fn finite_probe(input: &[f32], output: &mut [u32]) {
    use cubecl::prelude::*;
    let i = ABSOLUTE_POS;
    if i < 12 {
        let mut value = input[i];
        if i >= 9 {
            value = input[i] / input[4];
        }
        output[i] = u32::cast_from(finite::is_finite(value));
    }
}

#[test]
// The production helper must classify uploaded values and runtime device division.
fn finite_predicates_match_ieee_classification() {
    use cubecl::prelude::*;
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        0.0,
        1.0,
        -1.0,
    ];
    let client = gpu_client().unwrap();
    let input = client.create_from_slice(f32::as_bytes(&values));
    let output = client.empty(12 * size_of::<u32>());
    unsafe {
        finite_probe::launch_unchecked::<GpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(input, 12),
            BufferArg::from_raw_parts(output.clone(), 12),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let classifications = u32::from_bytes(&bytes);
    for (i, classification) in classifications.iter().enumerate() {
        let expected = (3..9).contains(&i);
        assert_eq!(
            *classification,
            u32::from(expected),
            "production bit classification at {i}"
        );
    }
}

#[cubecl::prelude::cube(launch_unchecked)]
fn small_angle_probe(input: &[f32], output: &mut [f32]) {
    use cubecl::prelude::*;
    let i = ABSOLUTE_POS;
    if i < input.len() {
        output[2 * i] = trig::sin(input[i]);
        output[2 * i + 1] = f32::cos(input[i]); // Native cosine is what ships.
    }
}

/// SE(2) divides sin(theta) by theta: absolute-error-only trig is insufficient.
#[test]
fn small_angle_trig_stays_within_two_ulps_of_the_cpu() {
    use cubecl::prelude::*;
    let mut values = Vec::new();
    for extent in [0.001f32, 0.5] {
        for i in 0..=1024 {
            values.push((i as f32 / 512.0 - 1.0) * extent);
        }
    }
    let client = gpu_client().unwrap();
    let input = client.create_from_slice(f32::as_bytes(&values));
    let output = client.empty(values.len() * 2 * size_of::<f32>());
    unsafe {
        small_angle_probe::launch_unchecked::<GpuRuntime>(
            &client,
            CubeCount::Static(values.len().div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(input, values.len()),
            BufferArg::from_raw_parts(output.clone(), values.len() * 2),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let actual = f32::from_bytes(&bytes);
    let mut worst = [0u32; 2];
    for (i, theta) in values.iter().enumerate() {
        for (operation, expected) in [theta.sin(), theta.cos()].iter().enumerate() {
            let measured = actual[2 * i + operation];
            assert!(measured.is_finite());
            // Same-sign floats have monotonic bit patterns. Zero signs are equivalent.
            let ulps = if measured == *expected {
                0
            } else {
                assert_eq!(measured.is_sign_negative(), expected.is_sign_negative());
                measured.to_bits().abs_diff(expected.to_bits())
            };
            worst[operation] = worst[operation].max(ulps);
            assert!(
                ulps <= 2,
                "theta={theta:e}, operation={operation}, GPU={measured:e}, CPU={expected:e}, ulps={ulps}"
            );
        }
    }
    println!("small-angle sin/cos maximum ULP errors: {worst:?}");
}
