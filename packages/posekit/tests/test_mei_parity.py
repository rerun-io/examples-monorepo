"""posekit's torch Mei camera against kornia-staging's Rust ``camera::Mei``, an independent implementation of the same model.

The reference pixels and rays below come from the Rust crate in float64 (``Mei::project`` / ``Mei::unproject``), printed
by a scratch binary with a path dependency on ``kornia-staging-3d``.
Three cameras: clip B lens 0 (``mei_v6``, the ``LENS0`` of ``test_sphere.py``), the X4 Air lens 0 ``V3`` factory string
as kornia-staging's tests round it (``mei_v3``, ``xi > 1``), and a small ``mei_v3`` lens with ``xi < 1`` and larger
tangential terms. ``None`` marks a ray or pixel the Rust camera rejects (outside its field).
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pytest
import torch

from posekit.ops.sphere import MeiCamera

ProjectCase: TypeAlias = tuple[tuple[float, float, float], tuple[float, float] | None]
"""A camera-frame ray and the Rust pixel, or ``None`` where Rust rejects the ray."""
UnprojectCase: TypeAlias = tuple[tuple[float, float], tuple[float, float, float] | None]
"""A pixel and the Rust unit ray, or ``None`` where Rust rejects the pixel."""

# Both sides evaluate the same float64 operations in the same order; at the crate commit the pixels agree bit for bit.
# 1e-9 px leaves room for another sqrt or norm reduction order and still catches a wrong term: zeroing the weakest
# coefficient (d12 of lens 0) moves some pixel here by 0.12 px, the weakest mei_v3 term (p2 of the X4 Air lens) by 1.3 px.
PIXEL_ATOL: float = 1e-9
# Rust stops Newton at a residual of 2e-15 (1 + |target|) normalized units with the analytic Jacobian; posekit runs 20
# fixed steps with a finite-difference Jacobian to the same root. The rays agree to 1.3e-15; 1e-12 on a unit ray is
# a few nanopixels at these focals.
RAY_ATOL: float = 1e-12

LENS0_V6: MeiCamera = MeiCamera.from_calibration(
    "mei_v6",
    1835.3690185546875,
    1835.0146484375,
    958.292236328125,
    965.8672485351562,
    [2.3149399757385254, 1.7453292608261108, 0.9386149644851685, -1.6840213537216187, 2.191660165786743, 3.8576207160949707, 0.0,
     -0.0049596200697124004, 0.0015607599634677172, -0.003307220060378313, 0.003974860068410635, 0.008278979919850826,
     -0.0012531500542536378, 0.024164050817489624, -0.0014354699524119496],
)  # fmt: skip
X4AIR_V3: MeiCamera = MeiCamera.from_calibration(
    "mei_v3", 1834.6, 1834.2, 960.242, 965.681, [2.31494, 1.7453292519943295, 0.96273077, -2.06431413, 4.31214333, 0.00172895, -0.00121632]
)
SMALL_V3: MeiCamera = MeiCamera.from_calibration("mei_v3", 400.0, 410.0, 320.0, 240.0, [0.8, 1.7453292519943295, 0.1, -0.05, 0.01, 0.02, -0.01])

# Rays: the axis; bearings at 5, 20, 40, 60, 75, 89, 95, 99.9 degrees off the axis (azimuths 0..310); a non-unit ray
# (3x the 50-degree bearing); 100.1 degrees (just past the 100-degree half field) and straight back.
LENS0_V6_PROJECT: list[ProjectCase] = [
    ((0.0, 0.0, 1.0), (958.292236328125, 965.8672485351563)),
    ((0.08715574274765817, 0.0, 0.9961946980917455), (1006.6258487685502, 965.8676418633768)),
    ((0.24184476264797528, 0.24184476264797522, 0.9396926207859084), (1096.0638363421926, 1103.5471165518632)),
    ((-0.4545194776720436, 0.45451947767204365, 0.766044443118978), (677.0718170908801, 1247.4044028559947)),
    ((-0.8137976813493737, -0.29619813272602374, 0.5000000000000001), (386.9218592871091, 757.7061317051925)),
    ((-1.7743769570680046e-16, -0.9659258262890683, 0.25881904510252074), (959.9097280066859, 205.15831945941977)),
    ((0.6426897100201723, -0.7659277708398715, 0.0174524064372836), (1531.7013333760974, 285.77218977576604)),
    ((0.862729915662821, 0.4980973490458727, -0.08715574274765824), (1771.8381783198035, 1434.190011539795)),
    ((-0.9256999644550196, 0.3369272329229088, -0.1719291002794096), (43.882259108907874, 1300.0388229590158)),
    ((0.5948006738224385, 2.219826335081544, 1.9283628290596182), (1088.836444116022, 1450.8620436184922)),
    ((0.8526047639644255, 0.4922515899872183, -0.17536672609198697), None),  # OutsideDomain
    ((1.2246467991473532e-16, 0.0, -1.0), None),  # OutsideDomain
]
LENS0_V6_UNPROJECT: list[UnprojectCase] = [
    ((958.292236328125, 965.8672485351563), (0.0, 0.0, 1.0)),
    ((1000.0, 900.0), (0.07498482159511126, -0.11847647300000479, 0.9901215086421606)),
    ((1300.5, 700.25), (0.5435120921731863, -0.4225239262640234, 0.7253055476114603)),
    ((400.0, 1500.0), (-0.7018126054311697, 0.6693526186276756, 0.24377477062687758)),
    ((150.0, 960.0), (-0.9835510291262016, -0.0073673614062995775, 0.1804801792183799)),
    ((1700.0, 400.0), (0.7913054574694481, -0.6063643055641987, -0.07847293747983608)),
    ((960.0, 1900.0), (-0.0010594599042301996, 0.9974439699176489, -0.07144511473594184)),
    ((1950.0, 965.0), None),  # OutsideDomain
]
X4AIR_V3_PROJECT: list[ProjectCase] = [
    ((0.0, 0.0, 1.0), (960.242, 965.681)),
    ((0.08715574274765817, 0.0, 0.9961946980917455), (1008.5598967850531, 965.6831971891362)),
    ((0.24184476264797528, 0.24184476264797522, 0.9396926207859084), (1097.9687756224491, 1103.43740523967)),
    ((-0.4545194776720436, 0.45451947767204365, 0.766044443118978), (678.8828525079643, 1247.0197291219542)),
    ((-0.8137976813493737, -0.29619813272602374, 0.5000000000000001), (389.008248489302, 758.1912654562002)),
    ((-1.7743769570680046e-16, -0.9659258262890683, 0.25881904510252074), (959.9277024312726, 205.20382239838193)),
    ((0.6426897100201723, -0.7659277708398715, 0.0174524064372836), (1532.164880097, 284.3326352386406)),
    ((0.862729915662821, 0.4980973490458727, -0.08715574274765824), (1772.7641827987336, 1435.580298803075)),
    ((-0.9256999644550196, 0.3369272329229088, -0.1719291002794096), (45.61505593468871, 1299.00393052952)),
    ((0.5948006738224385, 2.219826335081544, 1.9283628290596182), (1090.1180751537042, 1451.0506701066358)),
    ((0.8526047639644255, 0.4922515899872183, -0.17536672609198697), None),  # OutsideDomain
    ((1.2246467991473532e-16, 0.0, -1.0), None),  # OutsideDomain
]
X4AIR_V3_UNPROJECT: list[UnprojectCase] = [
    ((958.292236328125, 965.8672485351563), (-0.0035230038556969137, 0.00033659747052690363, 0.9999937375533787)),
    ((1000.0, 900.0), (0.07153072328837447, -0.11819431635301109, 0.9904107527726489)),
    ((1300.5, 700.25), (0.5410060734837717, -0.4222398176527362, 0.7273417111937523)),
    ((400.0, 1500.0), (-0.7024022338442362, 0.6698586401124214, 0.2406667907239477)),
    ((150.0, 960.0), (-0.9842980385981724, -0.007507663476156232, 0.17635477368338748)),
    ((1700.0, 400.0), (0.7919476313359196, -0.6060321951216463, -0.07445755634900575)),
    ((960.0, 1900.0), (0.0002129377978647488, 0.9973777479564048, -0.07237114438023529)),
    ((1950.0, 965.0), None),  # OutsideDomain
]
SMALL_V3_PROJECT: list[ProjectCase] = [
    ((0.0, 0.0, 1.0), (320.0, 240.0)),
    ((0.08715574274765817, 0.0, 0.9961946980917455), (339.38528570595287, 240.0193062999285)),
    ((0.24184476264797528, 0.24184476264797522, 0.9396926207859084), (375.81712171571195, 297.68795442256396)),
    ((-0.4545194776720436, 0.45451947767204365, 0.766044443118978), (199.41417187373554, 364.29120759203397)),
    ((-0.8137976813493737, -0.29619813272602374, 0.5000000000000001), (58.1071800162282, 146.59708596901737)),
    ((-1.7743769570680046e-16, -0.9659258262890683, 0.25881904510252074), (316.67107463190825, -133.88768375016798)),
    ((0.6426897100201723, -0.7659277708398715, 0.0174524064372836), (634.1537435846442, -138.79611138273646)),
    ((0.862729915662821, 0.4980973490458727, -0.08715574274765824), (836.3899796806649, 566.228810537704)),
    ((-0.9256999644550196, 0.3369272329229088, -0.1719291002794096), (-363.8319528519297, 511.6184577263953)),
    ((0.5948006738224385, 2.219826335081544, 1.9283628290596182), (376.1601228622627, 461.45741246988155)),
    ((0.8526047639644255, 0.4922515899872183, -0.17536672609198697), None),  # OutsideDomain
    ((1.2246467991473532e-16, 0.0, -1.0), None),  # OutsideDomain
]
SMALL_V3_UNPROJECT: list[UnprojectCase] = [
    ((320.0, 240.0), (0.0, 0.0, 1.0)),
    ((350.0, 200.0), (0.13396043507017083, -0.17444873459000565, 0.9755112715062579)),
    ((100.0, 400.0), (-0.6907668562795932, 0.48363402638676783, 0.5375307235745048)),
    ((600.0, 30.0), (0.7650902561507835, -0.5701552408516092, 0.2992656032232439)),
    ((-300.0, 240.0), (-0.9964106937213381, -0.025008685268465423, -0.08087209097645809)),
    ((320.0, 1100.0), None),  # OutsideDomain
]

CASES: list[tuple[MeiCamera, list[ProjectCase], list[UnprojectCase]]] = [
    (LENS0_V6, LENS0_V6_PROJECT, LENS0_V6_UNPROJECT),
    (X4AIR_V3, X4AIR_V3_PROJECT, X4AIR_V3_UNPROJECT),
    (SMALL_V3, SMALL_V3_PROJECT, SMALL_V3_UNPROJECT),
]
IDS: list[str] = ["lens0_mei_v6", "x4air_mei_v3", "small_xi_mei_v3"]


@pytest.mark.parametrize(("camera", "project_cases", "unproject_cases"), CASES, ids=IDS)
def test_project_matches_rust(camera: MeiCamera, project_cases: list[ProjectCase], unproject_cases: list[UnprojectCase]) -> None:
    rays: torch.Tensor = torch.tensor([ray for ray, _ in project_cases], dtype=torch.float64)
    pixels, valid = camera.project(rays)
    assert valid.tolist() == [pixel is not None for _, pixel in project_cases]
    expected: np.ndarray = np.array([pixel for _, pixel in project_cases if pixel is not None])
    np.testing.assert_allclose(pixels[valid].numpy(), expected, rtol=0.0, atol=PIXEL_ATOL)


@pytest.mark.parametrize(("camera", "project_cases", "unproject_cases"), CASES, ids=IDS)
def test_unproject_matches_rust(camera: MeiCamera, project_cases: list[ProjectCase], unproject_cases: list[UnprojectCase]) -> None:
    pixels: torch.Tensor = torch.tensor([pixel for pixel, _ in unproject_cases], dtype=torch.float64)
    rays, valid = camera.unproject(pixels)
    assert valid.tolist() == [ray is not None for _, ray in unproject_cases]
    expected: np.ndarray = np.array([ray for _, ray in unproject_cases if ray is not None])
    np.testing.assert_allclose(rays[valid].numpy(), expected, rtol=0.0, atol=RAY_ATOL)
