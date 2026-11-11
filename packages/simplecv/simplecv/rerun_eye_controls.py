"""Eye controls for 3D views, shared by every package: a hand-placed orbital eye, and the chase camera.

The Follow pattern of ``rerun-io/eye_control_example``: a view whose ``origin`` is the moving body (a rig node, a
tracked camera) and an eye fixed in that body's frame, so the eye rides the body, position and heading. In Rerun 0.42
an explicit ``rr.CoordinateFrame`` anywhere in the recording stops the eye from turning with the origin (it keeps the
world's orientation), so a recording with a Follow view logs no explicit frames.
"""

import rerun.blueprint as rrb

FOLLOW_BACK_M: float = 0.9
"""How far behind the device the follow eye sits, along the device's own forward."""
FOLLOW_UP_M: float = 0.45
"""How far above the device the follow eye sits, along the device's own up."""
FOLLOW_AHEAD_M: float = 0.3
"""How far ahead of the device the follow eye aims, so the shot leads the motion.

The three defaults are dataforge's headset tuning (LaMAria's R_01_easy, the MSD headsets): a shot that keeps the
camera frusta and the last ten seconds of trail in view without the ground filling it. Other devices pass their own."""


def eye_controls_from_pose(
    position: tuple[float, float, float],
    look_target: tuple[float, float, float],
    eye_up: tuple[float, float, float],
    *,
    spin_speed: float = 0.0,
) -> rrb.EyeControls3D:
    """An orbital eye placed by hand, in the coordinates of the view's origin.

    Orbital: the eye orbits ``look_target`` when dragged, which is what a person
    inspecting a rig or a scene wants, whereas first-person controls fly the eye and
    lose the subject. When the view's origin is a moving body, an eye fixed in that
    frame rides the body (the ``rerun-io/eye_control_example`` pattern).

    Args:
        position: Where the eye sits.
        look_target: What it aims at and orbits around.
        eye_up: Which way is up for the eye; it fixes the horizon's roll.
        spin_speed: How fast the eye orbits ``look_target`` by itself; 0 keeps it still.

    Returns:
        An orbital eye, still unless ``spin_speed`` is set.
    """
    # EyeControls3D is marked unstable by the SDK; re-validate this factory on Rerun bumps.
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=spin_speed,
    )


def follow_eye_controls(
    forward: tuple[float, float, float],
    up: tuple[float, float, float],
    *,
    back_m: float = FOLLOW_BACK_M,
    up_m: float = FOLLOW_UP_M,
    ahead_m: float = FOLLOW_AHEAD_M,
    aim_down_m: float = 0.0,
    side_m: float = 0.0,
    spin_speed: float = 0.0,
) -> rrb.EyeControls3D:
    """A chase camera derived from the device's own forward and up.

    It sits ``back_m`` behind the device and ``up_m`` above it and aims
    ``ahead_m`` in front of it, which keeps the device itself and the ground it
    moves over both in shot. Every direction comes from the device's frame
    because devices carry their rig frames at different orientations, so one
    hand-picked position cannot serve them all.

    Args:
        forward: Where the device looks, in the rig frame; unit length.
        up: The device's up, in the rig frame; unit length.
        back_m: Distance behind the device, along ``forward``.
        up_m: Distance above it, along ``up``.
        ahead_m: Distance in front of it that the eye aims at.
        aim_down_m: How far below the device the aim point sits (hands, for a headset).
        side_m: Offset to the device's right, for a three-quarter view.
        spin_speed: How fast the eye orbits its aim point by itself; 0 keeps it still.

    Returns:
        An orbital eye in the device's frame, for a view whose origin is the device.
    """
    right: tuple[float, float, float] = (
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    )
    position: tuple[float, float, float] = (
        -back_m * forward[0] + up_m * up[0] + side_m * right[0],
        -back_m * forward[1] + up_m * up[1] + side_m * right[1],
        -back_m * forward[2] + up_m * up[2] + side_m * right[2],
    )
    look_target: tuple[float, float, float] = (
        ahead_m * forward[0] - aim_down_m * up[0],
        ahead_m * forward[1] - aim_down_m * up[1],
        ahead_m * forward[2] - aim_down_m * up[2],
    )
    return eye_controls_from_pose(position, look_target, up, spin_speed=spin_speed)
