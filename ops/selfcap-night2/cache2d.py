"""Lossless Parquet cache of calibration-independent RTMW detector frames."""
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from serde import serde
from serde.json import from_json, to_json
from selfcap_curate.keypoints import Keypoints2D, ViewDetections


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CacheManifest:
    phones: dict[str, int]
    """Expected phone names and candidate counts."""
    frames: int
    """Number of frames shared by every view."""
    rate: float
    """Detector sample rate in Hz."""
    time_sha256: str
    """Identity of the shared float64 time grid."""


def save(directory: Path, detections: Keypoints2D) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{directory.name}-', dir=directory.parent))
    try:
        for phone, view in detections.views.items():
            n, p = view.boxes.shape[:2]
            columns = [pa.array(detections.times)]
            for values in (view.boxes, view.box_scores, view.xy, view.scores):
                columns.append(pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), int(values.size // n)))
            table = pa.Table.from_arrays(columns, names=['time_s', 'boxes', 'box_scores', 'xy', 'scores']).replace_schema_metadata(
                {b'phone': phone.encode(), b'persons': str(p).encode(), b'rate': str(detections.rate).encode()})
            pq.write_table(table, temporary / f'{phone}.parquet')
        manifest = CacheManifest({phone: view.boxes.shape[1] for phone, view in detections.views.items()},
                                 len(detections.times), detections.rate, hashlib.sha256(detections.times.tobytes()).hexdigest())
        (temporary / 'complete.json').write_text(to_json(manifest))
        load(temporary)  # Publish only a self-consistent, complete cache.
        if directory.exists():
            # The caller has rejected this old cache; never merge its partial files.
            shutil.rmtree(directory)
        temporary.replace(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load(directory: Path, *, expected_phones: tuple[str, ...] | None = None, expected_rate: float | None = None) -> Keypoints2D:
    marker = directory / 'complete.json'
    manifest = from_json(CacheManifest, marker.read_text()) if marker.exists() else None
    if manifest is None and expected_phones is None:
        raise ValueError('legacy detector cache requires the session phone set')
    views = {}
    times = None
    rate = None
    for path in sorted(directory.glob('*.parquet')):
        table = pq.read_table(path)
        meta = table.schema.metadata
        n, p = len(table), int(meta[b'persons'])
        phone = meta[b'phone'].decode()
        frame_times = table['time_s'].combine_chunks().to_numpy()
        frame_rate = float(meta[b'rate'])
        if phone in views or path.stem != phone:
            raise ValueError('duplicate or mismatched detector cache phone')
        if times is not None and (not np.array_equal(times, frame_times) or rate != frame_rate):
            raise ValueError('detector cache views have inconsistent grids or rates')
        times, rate = frame_times, frame_rate
        arrays = [table[name].combine_chunks().values.to_numpy() for name in ('boxes', 'box_scores', 'xy', 'scores')]
        views[phone] = ViewDetections(arrays[0].reshape(n, p, 4), arrays[1].reshape(n, p), arrays[2].reshape(n, p, 133, 2), arrays[3].reshape(n, p, 133))
    if times is None or not len(times) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('empty or invalid detector cache time grid')
    if expected_phones is not None and set(views) != set(expected_phones):
        raise ValueError('detector cache does not match session phones')
    if expected_rate is not None and rate != expected_rate:
        raise ValueError('detector cache does not match session rate')
    if manifest is not None and (manifest.phones != {phone: view.boxes.shape[1] for phone, view in views.items()}
                                 or manifest.frames != len(times) or manifest.rate != rate
                                 or manifest.time_sha256 != hashlib.sha256(times.tobytes()).hexdigest()):
        raise ValueError('incomplete detector cache or changed time grid')
    return Keypoints2D(times, views, rate)
