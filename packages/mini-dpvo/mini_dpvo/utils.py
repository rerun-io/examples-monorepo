from collections.abc import Generator
from types import TracebackType

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

all_times: list[float] = []


class Timer:
    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name: str = name
        self.enabled: bool = enabled

        if self.enabled:
            self.start: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
            self.end: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "Timer":
        if self.enabled:
            self.start.record()
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed: float = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(f"{self.name}: {elapsed:.2f}ms")


def coords_grid(b: int, n: int, h: int, w: int, **kwargs: object) -> Float[Tensor, "b n 2 h w"]:
    x: Float[Tensor, "w"] = torch.arange(0, w, dtype=torch.float, **kwargs)
    y: Float[Tensor, "h"] = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords: Float[Tensor, "2 h w"] = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    result: Float[Tensor, "b n 2 h w"] = coords[[1, 0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)
    return result


def coords_grid_with_index(
    d: Float[Tensor, "b n h w"], **kwargs: object
) -> tuple[Float[Tensor, "b n 3 h w"], Float[Tensor, "b n 1 h w"]]:
    b: int
    n: int
    h: int
    w: int
    b, n, h, w = d.shape
    i: Float[Tensor, "b n h w"] = torch.ones_like(d)
    x: Float[Tensor, "w"] = torch.arange(0, w, dtype=torch.float, **kwargs)
    y: Float[Tensor, "h"] = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y: Float[Tensor, "b n h w"] = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x: Float[Tensor, "b n h w"] = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords: Float[Tensor, "b n 3 h w"] = torch.stack([x, y, d], dim=2)
    index: Float[Tensor, "b n 1 h w"] = torch.arange(0, n, dtype=torch.float, **kwargs).view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index


def patchify(x: Float[Tensor, "b n c h w"], patch_size: int = 3) -> Float[Tensor, "b n_patches c ps ps"]:
    b: int
    n: int
    c: int
    h: int
    w: int
    b, n, c, h, w = x.shape
    x_reshaped: Float[Tensor, "bn c h w"] = x.view(b * n, c, h, w)
    y: Float[Tensor, "bn patches_times_c patch_elements"] = F.unfold(x_reshaped, patch_size)
    y_t: Float[Tensor, "bn n_patches patch_elements"] = y.transpose(1, 2)
    result: Float[Tensor, "b n_patches c ps ps"] = y_t.reshape(b, -1, c, patch_size, patch_size)
    return result


def pyramidify(fmap: Float[Tensor, "b n c h w"], lvls: list[int] = [1]) -> list[Float[Tensor, "b n c h_l w_l"]]:
    b: int
    n: int
    c: int
    h: int
    w: int
    b, n, c, h, w = fmap.shape

    pyramid: list[Float[Tensor, "b n c h_l w_l"]] = []
    for lvl in lvls:
        gmap: Float[Tensor, "bn c h_l w_l"] = F.avg_pool2d(fmap.view(b * n, c, h, w), lvl, stride=lvl)
        pyramid += [gmap.view(b, n, c, h // lvl, w // lvl)]

    return pyramid


def all_pairs_exclusive(n: int, **kwargs: object) -> tuple[Int[Tensor, "n_pairs"], Int[Tensor, "n_pairs"]]:
    ii: Int[Tensor, "n n"]
    jj: Int[Tensor, "n n"]
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k: torch.Tensor = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)


def set_depth(
    patches: Float[Tensor, "*batch 3 ps ps"], depth: Float[Tensor, "*batch"]
) -> Float[Tensor, "*batch 3 ps ps"]:
    patches[..., 2, :, :] = depth[..., None, None]
    return patches


def flatmeshgrid(*args: torch.Tensor, **kwargs: object) -> Generator[torch.Tensor, None, None]:
    grid: tuple[torch.Tensor, ...] = tuple(torch.meshgrid(*args, **kwargs))
    return (x.reshape(-1) for x in grid)
