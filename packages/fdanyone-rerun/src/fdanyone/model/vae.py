"""Per-view VAE execution and publication for generation stages."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from threading import BoundedSemaphore, Event
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from fdanyone.config import INFERENCE
from fdanyone.errors import FourDAnyoneError
from fdanyone.video import write_video

if TYPE_CHECKING:
    from taehv import TAEHV
    from torch import Tensor

    from fdanyone.vendor.diffsynth.models.wan_video_vae import WanVideoVAE38

LOGGER = logging.getLogger("fdanyone")

Output = TypeVar("Output")


def _bf16_autocast():
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _rgb_frames(video: Tensor) -> tuple[np.ndarray, ...]:
    """Freeze the released CUDA scaling boundary before CPU publication.

    The float-to-uint8 cast truncates toward zero on the device exactly as numpy's ``astype`` did on the host, so one
    uint8 [frames, H, W, 3] copy replaces a float32 copy four times its size and a per-frame host conversion.
    """

    import torch

    scaled = video.detach().float().add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
    frames = scaled.to(torch.uint8).permute(1, 2, 3, 0).contiguous().to(device="cpu").numpy()
    return tuple(frames[frame_index] for frame_index in range(frames.shape[0]))


PUBLISH_SLOTS: int = 6
"""Decoded views held for or in x264 at once (six: one denoising group)."""

_TINY_DECODERS: dict[tuple[str, str], TAEHV] = {}


def load_tiny_decoder(path: Path, device: str) -> TAEHV:
    """TAEW2.2 (Wan 2.2 5B latents: 48 channels, patch 2) in fp16 on ``device``, loaded once per process."""

    import torch
    from taehv import TAEHV

    key = (str(path), device)
    if key not in _TINY_DECODERS:
        decoder = TAEHV(checkpoint_path=str(path))
        if (decoder.latent_channels, decoder.patch_size) != (48, 2):
            raise FourDAnyoneError(f"{path} is not TAEW2.2 (latent_channels={decoder.latent_channels}).")
        _TINY_DECODERS[key] = decoder.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    return _TINY_DECODERS[key]


def _tiny_decode_view(decoder, latent: Tensor, device: str) -> Tensor:
    """Decode one view's normalized [C,F,H,W] latents to [1,3,frames,H,W] pixels in [-1, 1], as the Wan VAE returns."""

    import torch

    frames = decoder.decode_video(
        latent.unsqueeze(0).transpose(1, 2).to(device=device, dtype=torch.float16),
        parallel=False,
        show_progress_bar=False,
    )
    return frames.transpose(1, 2).float().mul_(2.0).sub_(1.0)


class VaeExecutor:
    """Run independent views on an ordered CUDA device pool.

    Each worker owns one model replica and processes a deterministic strided
    subset of camera indices. Models return to CPU between stages so the same
    replicas can be reused without competing with the DiT for device memory.
    """

    def __init__(self, model: WanVideoVAE38, devices: tuple[str, ...]) -> None:
        if not devices:
            raise FourDAnyoneError("VAE execution requires at least one CUDA device.")
        self.devices = devices
        self._models = [model]
        self.last_peak_vram_bytes: dict[str, dict[str, int]] = {}

    @classmethod
    def load(cls, path: str | Path, devices: tuple[str, ...]) -> VaeExecutor:
        from fdanyone.model.loader import load_vae

        return cls(load_vae(path), devices)

    @property
    def latent_channels(self) -> int:
        return int(self._models[0].model.z_dim)

    @property
    def upsampling_factor(self) -> int:
        return int(self._models[0].upsampling_factor)

    def _ensure_models(self, count: int) -> None:
        while len(self._models) < count:
            self._models.append(deepcopy(self._models[0]))

    @staticmethod
    def _worker_count(num_views: int, num_devices: int) -> int:
        if num_views <= 0:
            raise FourDAnyoneError("VAE execution requires at least one view.")
        return min(num_views, num_devices)

    @staticmethod
    def _worker_indices(num_views: int, worker_index: int, worker_count: int) -> tuple[int, ...]:
        return tuple(range(worker_index, num_views, worker_count))

    def _run_workers(
        self,
        *,
        num_views: int,
        operation: Callable[[int, str, tuple[int, ...], Event], None],
        stopped: Event | None = None,
    ) -> None:
        import torch

        worker_count = self._worker_count(num_views, len(self.devices))
        stopped = Event() if stopped is None else stopped
        peaks: dict[str, dict[str, int]] = {}

        def run(worker_index: int) -> None:
            device = self.devices[worker_index]
            device_index = int(device.removeprefix("cuda:"))
            indices = self._worker_indices(num_views, worker_index, worker_count)
            torch.cuda.set_device(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)
            try:
                operation(worker_index, device, indices, stopped)
                torch.cuda.synchronize(device_index)
                peaks[device] = {
                    "allocated": int(torch.cuda.max_memory_allocated(device_index)),
                    "reserved": int(torch.cuda.max_memory_reserved(device_index)),
                }
            except BaseException:
                stopped.set()
                raise

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vae") as pool:
            futures = [pool.submit(run, worker_index) for worker_index in range(worker_count)]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stopped.set()
                for future in futures:
                    future.cancel()
                raise
        # Do not let a completed worker release cached mappings while a peer is
        # still executing.  This raced with expandable segments in low-memory
        # multi-GPU inference.  Join every worker before serial cache cleanup.
        for worker_index in range(worker_count):
            torch.cuda.set_device(int(self.devices[worker_index].removeprefix("cuda:")))
            torch.cuda.empty_cache()
        self.last_peak_vram_bytes = peaks

    @contextmanager
    def _wan_on(self, worker_index: int, device: str) -> Iterator[WanVideoVAE38]:
        """Worker ``worker_index``'s Wan replica on ``device``, back on the CPU afterwards."""

        model = self._models[worker_index]
        model.to(device)
        try:
            yield model
        finally:
            model.to("cpu")

    @staticmethod
    def _encode_view(model: WanVideoVAE38, video: Tensor, device: str) -> Tensor:
        import torch

        if INFERENCE.tiled_vae:
            tile_size = tuple(size * model.upsampling_factor for size in INFERENCE.vae_tile_size)
            tile_stride = tuple(stride * model.upsampling_factor for stride in INFERENCE.vae_tile_stride)
            batched = video.unsqueeze(0).to(dtype=torch.bfloat16, device="cpu")
            return model.tiled_encode(batched, device, tile_size, tile_stride)
        batched = video.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        return model.encode_view(batched)

    @staticmethod
    def _decode_view(model: WanVideoVAE38, latent: Tensor, device: str) -> Tensor:
        import torch

        if INFERENCE.tiled_vae:
            batched = latent.unsqueeze(0).to(dtype=torch.bfloat16, device="cpu")
            return model.tiled_decode(
                batched,
                device,
                INFERENCE.vae_tile_size,
                INFERENCE.vae_tile_stride,
            )
        batched = latent.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        return model.decode_view(batched)

    def encode(self, videos: Tensor) -> Tensor:
        """Encode CPU-resident views and gather latents by input index."""

        import torch

        if videos.ndim != 5 or videos.device.type != "cpu":
            raise FourDAnyoneError(f"VAE input must be CPU [V,C,F,H,W], got {tuple(videos.shape)} on {videos.device}.")
        outputs: list[Tensor | None] = [None] * videos.shape[0]

        def operation(worker_index: int, device: str, indices: tuple[int, ...], stopped: Event) -> None:
            with self._wan_on(worker_index, device) as model, torch.inference_mode(), _bf16_autocast():
                for index in indices:
                    if stopped.is_set():
                        return
                    encoded = self._encode_view(model, videos[index], device)
                    outputs[index] = encoded[0].detach().to(device="cpu").contiguous()
                    del encoded

        self._ensure_models(self._worker_count(videos.shape[0], len(self.devices)))
        self._run_workers(num_views=videos.shape[0], operation=operation)
        if any(output is None for output in outputs):
            raise RuntimeError("VAE encode completed without every canonical view.")
        return torch.stack(outputs)  # type: ignore[arg-type]

    def _decode_and_publish(
        self,
        latents: Tensor,
        sink: Callable[[int, tuple[np.ndarray, ...]], Output],
        tiny_decoder: Path | None = None,
    ) -> tuple[Output, ...]:
        import torch

        if latents.ndim != 5 or latents.device.type != "cpu":
            raise FourDAnyoneError(
                f"VAE latents must be CPU [V,C,F,H,W], got {tuple(latents.shape)} on {latents.device}."
            )
        num_views = int(latents.shape[0])
        worker_count = self._worker_count(num_views, len(self.devices))
        # Decoding a view takes ~1 s and its x264 encode ~6 s: up to PUBLISH_SLOTS views are encoded at once (one view's
        # frames are ~330 MB of host memory), and a worker decodes its next view while earlier ones are in x264.
        encode_slots = max(worker_count + 1, PUBLISH_SLOTS)
        slots = BoundedSemaphore(encode_slots)
        sink_futures: list[Future[Output] | None] = [None] * num_views
        stopped = Event()

        # The decoder is chosen once: TAEW2.2 workers never touch the Wan VAE (it stays on the CPU for source encode).
        @contextmanager
        def decoder_on(worker_index: int, device: str) -> Iterator[Callable[[Tensor], Tensor]]:
            if tiny_decoder is not None:
                taehv = load_tiny_decoder(tiny_decoder, device)
                yield lambda latent: _tiny_decode_view(taehv, latent, device)
                return
            with self._wan_on(worker_index, device) as model:
                yield lambda latent: self._decode_view(model, latent, device)

        if tiny_decoder is None:
            self._ensure_models(worker_count)

        with ThreadPoolExecutor(max_workers=encode_slots, thread_name_prefix="video-codec") as codec_pool:

            def release_slot(future: Future[Output]) -> None:
                if future.exception() is not None:
                    stopped.set()
                slots.release()

            def operation(
                worker_index: int,
                device: str,
                indices: tuple[int, ...],
                worker_stop: Event,
            ) -> None:
                with decoder_on(worker_index, device) as decode, torch.inference_mode(), _bf16_autocast():
                    for index in indices:
                        if worker_stop.is_set():
                            return
                        slots.acquire()
                        try:
                            decoded = decode(latents[index])
                            device_video = decoded[0].detach()
                            rgb_frames = _rgb_frames(device_video)
                            del decoded, device_video
                            future = codec_pool.submit(
                                sink,
                                index,
                                rgb_frames,
                            )
                            del rgb_frames
                            future.add_done_callback(release_slot)
                            sink_futures[index] = future
                        except BaseException:
                            slots.release()
                            stopped.set()
                            raise

            self._run_workers(num_views=num_views, operation=operation, stopped=stopped)
            for future in sink_futures:
                if future is not None and future.done() and future.exception() is not None:
                    future.result()
            if any(future is None for future in sink_futures):
                raise RuntimeError("VAE decode completed without every canonical view.")
            return tuple(future.result() for future in sink_futures if future is not None)

    def publish_targets(
        self, latents: Tensor, output_dir: Path, fps: Fraction, tiny_decoder: Path | None = None
    ) -> tuple[Path, ...]:
        video_root = output_dir / "videos"
        video_root.mkdir(parents=True, exist_ok=False)

        def publish(camera_id: int, rgb_frames: tuple[np.ndarray, ...]) -> Path:
            LOGGER.info("Publishing target camera %02d", camera_id)
            return write_video(
                iter(rgb_frames),
                video_root / f"{camera_id:02d}.mp4",
                fps,
                crf=INFERENCE.target_h264_crf,
                preset=INFERENCE.h264_preset,
            )

        return self._decode_and_publish(latents, publish, tiny_decoder=tiny_decoder)

    def release_replicas(self) -> None:
        """Keep one CPU model while releasing stage-local parallel replicas."""

        del self._models[1:]
        gc.collect()

    def close(self) -> None:
        """Release all CPU replicas after the final VAE stage."""

        self._models.clear()
        gc.collect()
