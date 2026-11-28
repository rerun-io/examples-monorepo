"""compute_clip_priors' pipelined postprocess (worker thread + second CUDA stream) gives the maps of the plain loop."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monopriors", reason="monopriors is in the open4dgs envs")


class _GraphMoGe:
    """A MoGe stand-in whose network outputs depend on the frames; ``forward_geometry`` reuses one output buffer per call
    (as the TensorRT engine does), so a postprocess that read the alias instead of a copy would see the next batch."""

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None

    def forward_geometry(self, rgb: torch.Tensor):  # noqa: ANN201
        from monopriors.models.moge_v2.predictor import MoGeV2GeometryGraphOutput

        b, h, w, _ = rgb.shape
        nh, nw = h // 2, w // 2
        x = torch.nn.functional.interpolate(rgb.permute(0, 3, 1, 2).float() / 255.0, size=(nh, nw), mode="area").permute(0, 2, 3, 1)
        if self.buffer is None or self.buffer.shape[0] < b:
            self.buffer = torch.empty((b, nh, nw, 3), device=rgb.device)
        u = torch.linspace(-0.5, 0.5, nw, device=rgb.device)[None, None, :].expand(b, nh, nw)
        v = torch.linspace(-0.8, 0.8, nh, device=rgb.device)[None, :, None].expand(b, nh, nw)
        z = 1.0 + x[..., 0]
        self.buffer[:b] = torch.stack([u * z, v * z, z], dim=-1)
        normals = torch.nn.functional.normalize(x - 0.5 + 1e-3, dim=-1)
        return MoGeV2GeometryGraphOutput(points_bhw3=self.buffer[:b], normals_bhw3=normals, mask_bhw=x[..., 1], metric_scale_b=1.0 + x.mean(dim=(1, 2, 3)))

    def predict_geometry(self, rgb: torch.Tensor, *, fov_x: float | None = None):  # noqa: ANN201
        from monopriors.models.moge_v2.predictor import postprocess_moge_v2_geometry

        return postprocess_moge_v2_geometry(self.forward_geometry(rgb), (rgb.shape[1], rgb.shape[2]), fov_x=fov_x)


@pytest.mark.integration
def test_pipelined_priors_equal_the_plain_loop() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from open4dgs.catalog_clip import MultiViewClip
    from open4dgs.prior import encode_normals
    from open4dgs.prior_layer import compute_clip_priors

    gen = torch.Generator().manual_seed(0)
    frames = tuple(torch.randint(0, 255, (11, 64, 48, 3), dtype=torch.uint8, generator=gen) for _ in range(2))
    K = torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]]).expand(2, 3, 3).clone()
    clip = MultiViewClip(
        camera_names=("a", "b"),
        frames=frames,
        valid=tuple(torch.ones(64, 48, dtype=torch.bool) for _ in frames),
        K=K,
        world_T_cam=torch.eye(4).expand(2, 4, 4).clone(),
        fps=25.0,
        frame_indices=torch.arange(11),
        people=None,
    )
    model = _GraphMoGe()
    got = compute_clip_priors(clip, ("b", "a"), model, batch_size=4, device=torch.device("cuda"))
    assert got.camera_names == ("b", "a")
    for name in ("b", "a"):
        cam = clip.camera_names.index(name)
        fov_x = math.degrees(2.0 * math.atan(48 / (2.0 * 40.0)))
        depth, normals = [], []
        for start in range(0, 11, 4):
            out = model.predict_geometry(frames[cam][start : start + 4].cuda(), fov_x=fov_x)
            valid = (out.mask_bhw > 0.5) & (out.depth_bhw > 0)
            depth.append(torch.where(valid, out.depth_bhw, 0.0).half().cpu())
            normals.append(encode_normals(out.normals_bhw3, valid).cpu())
        index = got.camera_names.index(name)
        assert torch.equal(got.depth[index], torch.cat(depth))
        assert torch.equal(got.normals[index], torch.cat(normals))
