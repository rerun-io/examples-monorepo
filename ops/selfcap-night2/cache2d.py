"""Lossless Parquet cache of calibration-independent RTMW detector frames."""
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from selfcap_curate.keypoints import Keypoints2D, ViewDetections


def save(directory: Path, detections: Keypoints2D) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    for phone,view in detections.views.items():
        n,p=view.boxes.shape[:2]
        columns=[pa.array(detections.times)]
        for values in (view.boxes,view.box_scores,view.xy,view.scores):
            columns.append(pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)),int(values.size//n)))
        table=pa.Table.from_arrays(columns,names=['time_s','boxes','box_scores','xy','scores']).replace_schema_metadata({b'phone':phone.encode(),b'persons':str(p).encode(),b'rate':str(detections.rate).encode()})
        pq.write_table(table,directory/f'{phone}.parquet')


def load(directory: Path) -> Keypoints2D:
    views={};times=None;rate=None
    for path in sorted(directory.glob('*.parquet')):
        table=pq.read_table(path); meta=table.schema.metadata;n=len(table);p=int(meta[b'persons'])
        times=table['time_s'].combine_chunks().to_numpy();rate=float(meta[b'rate'])
        arrays=[table[name].combine_chunks().values.to_numpy() for name in ('boxes','box_scores','xy','scores')]
        views[meta[b'phone'].decode()]=ViewDetections(arrays[0].reshape(n,p,4),arrays[1].reshape(n,p),arrays[2].reshape(n,p,133,2),arrays[3].reshape(n,p,133))
    if times is None:raise ValueError('empty detector cache')
    return Keypoints2D(times,views,rate)
