try:
    import mast3r_slam_mojo_backends as _mojo_low
    from mast3r_slam import mojo_gn as _gn_backends  # pyrefly: ignore
except ImportError:
    from mast3r_slam import _backends as _gn_backends  # pyrefly: ignore
    _mojo_low = _gn_backends


gauss_newton_points = _gn_backends.gauss_newton_points
gauss_newton_points_step = getattr(_mojo_low, "gauss_newton_points_step", None) or _gn_backends.gauss_newton_points_step
gauss_newton_rays = _gn_backends.gauss_newton_rays
gauss_newton_rays_step = getattr(_mojo_low, "gauss_newton_rays_step", None) or _gn_backends.gauss_newton_rays_step
gauss_newton_calib = _gn_backends.gauss_newton_calib
gauss_newton_calib_step = getattr(_mojo_low, "gauss_newton_calib_step", None) or _gn_backends.gauss_newton_calib_step
pose_retr = getattr(_gn_backends, "pose_retr", None)
