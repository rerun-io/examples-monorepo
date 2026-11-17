"""Turn the output of `basalt_brief_dump.cpp` into the golden `.rrd` the kornia-staging golden tests read.

The dump's text is an intermediate: run the harness, pipe it here, keep only the `.rrd`. Every array keeps the dump's
values exactly: positions are the f32 Basalt received (printed with 9 digits), angles, weights and scores are f64
(printed with 17 digits), descriptors are Basalt's `std::bitset<256>` as four u64 words.

Layout, all static (AnyValues components):
- `/images/<k>`: `width`, `height` (u32); per keypoint `x`, `y` (f32), `described` (bool), `angle` (f64, 0 when not
  described), `descriptor` (u64, four words per keypoint, zeros when not described).
- `/matches/<m>`: `image_a`, `image_b` (u32); Basalt's `matchDescriptors` pairs as `index_a`, `index_b` (u32).
- `/bags/<i>`: `frame` (i64), `cam` (u32), `word` (u32), `weight` (f64) of one HashBoW bag.
- `/queries/<i>`: `frame`, `cam`; every hit as `all_frame`, `all_cam`, `all_score`; the top 10 older than
  `frame - 5` as `top_frame`, `top_cam`, `top_score`, in Basalt's order.
The recording property `/__properties/basalt_brief` names the harness and the RootSLAM commit.

Run: `pixi run -e slam-rs-dev python packages/kornia-staging/fixtures/basalt_brief/basalt_brief_to_rrd.py \
--dump basalt_brief.txt --out basalt_brief.rrd`
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import tyro


@dataclass(frozen=True, slots=True)
class Config:
    dump: Path
    """Text output of `basalt_brief_dump`."""
    out: Path
    """The `.rrd` to write."""
    rootslam_commit: str = "839629d"
    """RootSLAM commit (branch pixi) the harness was built against."""


def _hits(fields: list[str]) -> tuple[int, int, list[tuple[int, int, float]]]:
    hits = [(int(f), int(c), float(s)) for f, c, s in (h.split(",") for h in fields[4:])]
    assert len(hits) == int(fields[3])
    return int(fields[1]), int(fields[2]), hits


def _f32(text: str) -> np.float32:
    value = np.float32(text)
    # The dump prints f32 with 9 significant digits; the parse must give back the same f32.
    assert np.float32(float(f"{value:.9g}")) == value and float(f"{value:.9g}") == float(text), text
    return value


def main(config: Config) -> None:
    images: list[dict[str, list]] = []
    sizes: list[tuple[int, int]] = []
    matches: list[tuple[int, int, list[tuple[int, int]]]] = []
    bags: list[tuple[int, int, list[tuple[int, float]]]] = []
    queries: list[dict[str, object]] = []
    for line in config.dump.read_text().splitlines():
        fields = line.split(" ")
        match fields[0]:
            case "image":
                assert int(fields[1]) == len(images)
                sizes.append((int(fields[2]), int(fields[3])))
                images.append({"x": [], "y": [], "described": [], "angle": [], "descriptor": []})
            case "kp":
                image = images[-1]
                image["x"].append(_f32(fields[1]))
                image["y"].append(_f32(fields[2]))
                described = fields[3] != "out"
                image["described"].append(described)
                image["angle"].append(float(fields[3]) if described else 0.0)
                image["descriptor"].extend([int(w, 16) for w in fields[4:8]] if described else [0] * 4)
            case "match":
                pairs = [(int(a), int(b)) for a, b in (p.split(":") for p in fields[4:])]
                assert len(pairs) == int(fields[3])
                matches.append((int(fields[1]), int(fields[2]), pairs))
            case "bag":
                words = [(int(w), float(x)) for w, x in (p.split(":") for p in fields[4:])]
                assert len(words) == int(fields[3])
                bags.append((int(fields[1]), int(fields[2]), words))
            case "query_all":
                frame, cam, hits = _hits(fields)
                queries.append({"frame": frame, "cam": cam, "all": hits, "top": []})
            case "query_top":
                frame, cam, hits = _hits(fields)
                assert (queries[-1]["frame"], queries[-1]["cam"]) == (frame, cam)
                queries[-1]["top"] = hits
            case other:
                raise ValueError(f"{config.dump}: unknown record {other!r}")

    rr.init("kornia_staging_basalt_brief", recording_id="basalt-brief-golden")
    rr.save(config.out)
    rr.send_property(
        "basalt_brief",
        rr.AnyValues(
            harness="packages/kornia-staging/fixtures/basalt_brief/basalt_brief_dump.cpp",
            rootslam_commit=config.rootslam_commit,
            inputs="fixtures/frames/frame_00{0,1,2}_cam{0,1}.pgm widened << 8, then the synthetic 320x240 image",
        ),
    )
    for k, (image, (width, height)) in enumerate(zip(images, sizes, strict=True)):
        rr.log(
            f"images/{k}",
            rr.AnyValues(
                width=np.array([width], np.uint32),
                height=np.array([height], np.uint32),
                x=np.array(image["x"], np.float32),
                y=np.array(image["y"], np.float32),
                described=np.array(image["described"], np.bool_),
                angle=np.array(image["angle"], np.float64),
                descriptor=np.array(image["descriptor"], np.uint64),
            ),
            static=True,
        )
    for m, (a, b, pairs) in enumerate(matches):
        rr.log(
            f"matches/{m}",
            rr.AnyValues(
                image_a=np.array([a], np.uint32),
                image_b=np.array([b], np.uint32),
                index_a=np.array([p[0] for p in pairs], np.uint32),
                index_b=np.array([p[1] for p in pairs], np.uint32),
            ),
            static=True,
        )
    for i, (frame, cam, words) in enumerate(bags):
        rr.log(
            f"bags/{i}",
            rr.AnyValues(
                frame=np.array([frame], np.int64),
                cam=np.array([cam], np.uint32),
                word=np.array([w for w, _ in words], np.uint32),
                weight=np.array([x for _, x in words], np.float64),
            ),
            static=True,
        )
    for i, query in enumerate(queries):
        columns: dict[str, np.ndarray] = {
            "frame": np.array([query["frame"]], np.int64),
            "cam": np.array([query["cam"]], np.uint32),
        }
        for kind in ("all", "top"):
            hits: list[tuple[int, int, float]] = query[kind]  # type: ignore[assignment]
            columns[f"{kind}_frame"] = np.array([h[0] for h in hits], np.int64)
            columns[f"{kind}_cam"] = np.array([h[1] for h in hits], np.uint32)
            columns[f"{kind}_score"] = np.array([h[2] for h in hits], np.float64)
        rr.log(f"queries/{i}", rr.AnyValues(**columns), static=True)
    rr.get_data_recording().flush()
    print(f"{config.out}: {len(images)} images, {len(matches)} match sets, {len(bags)} bags, {len(queries)} queries")


if __name__ == "__main__":
    main(tyro.cli(Config))
