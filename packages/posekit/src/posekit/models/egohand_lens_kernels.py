# pyrefly: ignore-errors
"""Warp DSL lens kernels, compiled and type checked by Warp itself.

Python type checkers cannot interpret Warp's array-constructor annotations or
launch-dependent ``tid`` arity. The module directive applies only to these DSL
kernels; allocation, capture and public tensor APIs stay checked.
"""

import warp as wp

from egohand_inference.camera.camera_models import _project_rational_ray


@wp.func
def _fisheye_project(ray: wp.vec3d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), radial_count: int):
    radius = wp.sqrt(ray[0] * ray[0] + ray[1] * ray[1])
    theta = wp.atan2(radius, ray[2])
    power = theta * theta
    radial = wp.float64(1.0)
    for i in range(radial_count):
        radial += coefficients[i] * power
        power *= theta * theta
    factor = wp.float64(0.0)
    if radius > wp.float64(0.0):
        factor = theta * radial / radius
    x, y = ray[0] * factor, ray[1] * factor
    r2 = x * x + y * y
    if radial_count == 6:
        p0, p1 = coefficients[6], coefficients[7]
        xd = x + wp.float64(2.0) * p1 * x * y + p0 * (r2 + wp.float64(2.0) * x * x) + coefficients[8] * r2 + coefficients[9] * r2 * r2
        yd = y + wp.float64(2.0) * p0 * x * y + p1 * (r2 + wp.float64(2.0) * y * y) + coefficients[10] * r2 + coefficients[11] * r2 * r2
        x, y = xd, yd
    if ray[2] <= wp.float64(0.0):
        return wp.vec2d(wp.float64(wp.nan), wp.float64(wp.nan))
    return wp.vec2d(x * intrinsic[0] + intrinsic[2], y * intrinsic[1] + intrinsic[3])


@wp.func
def _project(ray: wp.vec3d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int):
    if radial_count == 0:
        return _project_rational_ray(ray, intrinsic, coefficients, tilt, radius_bound)
    return _fisheye_project(ray, intrinsic, coefficients, radial_count)


@wp.func
def _inverse(pixel: wp.vec2d, intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int):
    x = (pixel[0] - intrinsic[2]) / intrinsic[0]
    y = (pixel[1] - intrinsic[3]) / intrinsic[1]
    r2 = x * x + y * y
    if radial_count == 0 and wp.isfinite(radius_bound) and r2 >= radius_bound:
        scale = wp.sqrt(wp.float64(0.9) * radius_bound / r2)
        x *= scale
        y *= scale
    epsilon = wp.float64(1e-5)
    error = wp.float64(wp.inf)
    for _ in range(50):
        ray = wp.vec3d(x, y, wp.float64(1.0))
        projected = _project(ray, intrinsic, coefficients, tilt, radius_bound, radial_count)
        residual = projected - pixel
        error = wp.max(wp.abs(residual[0]), wp.abs(residual[1]))
        if not wp.isfinite(error) or error < wp.float64(1e-8):
            break
        dx = (_project(ray + wp.vec3d(epsilon, wp.float64(0.0), wp.float64(0.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - projected) / epsilon
        dy = (_project(ray + wp.vec3d(wp.float64(0.0), epsilon, wp.float64(0.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - projected) / epsilon
        determinant = dx[0] * dy[1] - dy[0] * dx[1]
        if not wp.isfinite(determinant) or determinant <= wp.float64(0.0):
            break
        step_x = (dy[1] * residual[0] - dy[0] * residual[1]) / determinant
        step_y = (dx[0] * residual[1] - dx[1] * residual[0]) / determinant
        factor = wp.float64(1.0)
        for _ in range(20):
            candidate_x, candidate_y = x - factor * step_x, y - factor * step_y
            trial = _project(wp.vec3d(candidate_x, candidate_y, wp.float64(1.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - pixel
            trial_error = wp.max(wp.abs(trial[0]), wp.abs(trial[1]))
            if wp.isfinite(trial_error) and trial_error < error:
                x, y = candidate_x, candidate_y
                break
            factor *= wp.float64(0.5)
    check = _project(wp.vec3d(x, y, wp.float64(1.0)), intrinsic, coefficients, tilt, radius_bound, radial_count) - pixel
    if not wp.isfinite(check[0]) or not wp.isfinite(check[1]) or wp.max(wp.abs(check[0]), wp.abs(check[1])) > wp.float64(1e-6):
        return wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    if radial_count > 0:
        # The released inverse rejects radial coordinates beyond the polynomial
        # at pi/2, even when a non-monotonic polynomial admits another solution.
        # Accepting that solution expands edge crops and changes predicted depth.
        theta = wp.atan(wp.sqrt(x * x + y * y))
        half_pi = wp.float64(1.5707963267948966)
        radial = wp.float64(1.0)
        maximum_radial = wp.float64(1.0)
        power = theta * theta
        maximum_power = half_pi * half_pi
        for i in range(radial_count):
            radial += coefficients[i] * power
            maximum_radial += coefficients[i] * maximum_power
            power *= theta * theta
            maximum_power *= half_pi * half_pi
        if theta * radial >= half_pi * maximum_radial:
            return wp.vec3d(wp.float64(wp.nan), wp.float64(wp.nan), wp.float64(wp.nan))
    return wp.normalize(wp.vec3d(x, y, wp.float64(1.0)))


@wp.kernel(enable_backward=False)
def unproject_pixels(pixels: wp.array2d(dtype=wp.vec2d), intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d,
                      radius_bound: wp.float64, radial_count: int, rays: wp.array2d(dtype=wp.vec3d)):
    hand, point = wp.tid()
    rays[hand, point] = _inverse(pixels[hand, point], intrinsic, coefficients, tilt, radius_bound, radial_count)


@wp.kernel(enable_backward=False)
def remap(focal: wp.array(dtype=wp.float64), rotation: wp.array(dtype=wp.mat33d), mirrored: wp.array(dtype=wp.bool),
           intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d, radius_bound: wp.float64, radial_count: int,
           pixels: wp.array3d(dtype=wp.vec2)):
    hand, row, col = wp.tid()
    u = col
    if mirrored[hand]:
        u = 255 - col
    ray = rotation[hand] @ wp.vec3d((wp.float64(u) - wp.float64(127.5)) / wp.float64(focal[hand]),
                                   (wp.float64(row) - wp.float64(127.5)) / wp.float64(focal[hand]), wp.float64(1.0))
    uv = _project(ray, intrinsic, coefficients, tilt, radius_bound, radial_count)
    if wp.isfinite(uv[0]) and wp.isfinite(uv[1]):
        pixels[hand, row, col] = wp.vec2(wp.float32(uv[0]), wp.float32(uv[1]))
    else:
        pixels[hand, row, col] = wp.vec2(-1.0, -1.0)


@wp.kernel(enable_backward=False)
def project_joints(points: wp.array2d(dtype=wp.vec3), intrinsic: wp.vec4d, coefficients: wp.array(dtype=wp.float64), tilt: wp.vec4d,
                    radius_bound: wp.float64, radial_count: int, pixels: wp.array2d(dtype=wp.vec2)):
    hand, joint = wp.tid()
    p = points[hand, joint]
    uv = _project(wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2])), intrinsic, coefficients, tilt, radius_bound, radial_count)
    pixels[hand, joint] = wp.vec2(wp.float32(uv[0]), wp.float32(uv[1]))
