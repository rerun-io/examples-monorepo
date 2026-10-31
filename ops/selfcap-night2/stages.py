"""GPU stage boundaries for the resumable overnight runner."""
import gc
import hashlib
import json
import resource
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
import rerun as rr
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from rerun.server import Server
from serde.json import from_json
from selfcap_curate.apis import align, calibrate, keypoints
import selfcap_curate.keypoints as keypoint_geometry
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.catalog_io import DATASET_NAME, native_phone_rate, open_segment, read_time_sync
from selfcap_curate.exo_landmarks import Landmarks, write_landmarks
from selfcap_curate.exo_refine import Refinement
from selfcap_curate.hands_gpu import pose_models, release_cuda_memory
from selfcap_curate.keypoints_gpu import keypoint_models
from selfcap_curate.oak_egohand import egohand_models
from selfcap_curate.quest_fit import read_session_episode

ROOT=Path('/tmp/selfcap-night2')
CATALOG='rerun+http://dgx-spark:51235'
DATASET='selfcap-post-ep1-v2'
REMOTE='/home/pablo/dataforge-dev/selfcap-post-ep1-v2'
ASSETS=Path('/tmp/egohand')


def release(failed: bool=False) -> None:
    if failed:
        pose_models.cache_clear()
        keypoint_models.cache_clear()
        egohand_models.cache_clear()
    gc.collect()
    release_cuda_memory(models=False)


def root(segment: str) -> Path:
    return ROOT/'runs'/segment


def private(segment: str):
    work=root(segment)/'s1'
    server=Server(host='127.0.0.1',port=0,datasets={DATASET_NAME:[work/'align'/f'{segment}.rrd']})
    return server


def s1(segment: str) -> dict:
    work=root(segment)/'s1'; work.mkdir(parents=True,exist_ok=True)
    if (work/'complete.json').exists():
        result=json.loads((work/'complete.json').read_text());result.update(raw_rtmw(segment));return result
    base=work/'base';base.mkdir(exist_ok=True)
    metrics={}; started=time.perf_counter()
    subprocess.run(['scp','-q',f'spark:/home/pablo/dataforge-dev/rrd/base/{segment}.rrd',str(base/f'{segment}.rrd')],check=True,timeout=180)
    metrics['fetch_s']=time.perf_counter()-started
    videos={}; started=time.perf_counter()
    if (work/'align').exists():shutil.rmtree(work/'align')
    align.main(align.Config(base_dir=base,output_dir=work/'align',segment_ids=(segment,),catalog_url=CATALOG),session_videos=videos)
    metrics['align_s']=time.perf_counter()-started
    release()
    with private(segment) as server:
        dataset=server.client().get_dataset(DATASET_NAME)
        dataset.register([(work/'align/sapiens2'/f'{segment}.rrd').as_uri()],layer_name='sapiens2').wait()
        started=time.perf_counter()
        if (work/'calib').exists():shutil.rmtree(work/'calib')
        calibrate.main(calibrate.Config(output_dir=work/'calib',segment_ids=(segment,),catalog_url=server.url(),donor_catalog_url=CATALOG,oak_egohand_assets=ASSETS,retain_models=True),session_videos=videos)
        metrics['calibrate_s']=time.perf_counter()-started
        metrics['rate']=native_phone_rate(videos[segment])
    metrics['gpu_peak_mib']=torch.cuda.max_memory_allocated()/2**20
    (work/'complete.json').write_text(json.dumps(metrics))
    metrics.update(raw_rtmw(segment,videos))
    return metrics


def ego_geometry_signature(path: Path) -> str:
    """Exact camera geometry on rig_04/05, independent of chunk partitioning."""
    rows=[]
    for chunk in RrdReader(path).stream():
        entity=str(chunk.entity_path)
        if not entity.startswith(('/world/rig_04','/world/rig_05')):continue
        batch=chunk.to_record_batch()
        stamps=[None]*len(batch) if chunk.is_static else batch['video_time'].cast('int64').to_pylist()
        for name in batch.schema.names:
            if name.startswith(('Transform3D:','Pinhole:')):
                rows.extend((entity,name,t,value) for t,value in zip(stamps,batch[name].to_pylist()))
    return hashlib.sha256(json.dumps(sorted(rows,key=lambda x:(x[0],x[1],-1 if x[2] is None else x[2])),sort_keys=True).encode()).hexdigest()


def s2(segment: str, decision: dict | None=None) -> dict:
    decision=decision or {}; work=root(segment); out=work/'s2'
    if out.exists():shutil.rmtree(out)
    out.mkdir(); final=out/'calib';final.mkdir(); source=work/'s1/calib'/f'{segment}.rrd'
    started=time.perf_counter()
    refined=bool(decision.get('refined',False))
    with private(segment) as server:
        dataset=server.client().get_dataset(DATASET_NAME)
        if refined:
            state_path=next((work/'s1/calib').glob('*.json'))
            state=from_json(CalibSession,Path(decision['calib_path']).read_text())
            record=open_segment(dataset,segment)
            write_calib_layer(final/f'{segment}.rrd',state,read_session_episode(record,read_time_sync(dataset,segment)))
        else:shutil.copy2(source,final/f'{segment}.rrd')
        current=final/f'{segment}.rrd'
        fallback=None
        if refined and ego_geometry_signature(current)!=ego_geometry_signature(source):
            fallback='Refinement changed rig_04/05 transform or pinhole; retained S1 geometry'
            print('EXO_REFINEMENT_REJECTED',segment,fallback,flush=True)
            refined=False
            shutil.copy2(source,current)
        reader=RrdReader(current); store=reader.recordings()[0]
        enriched=out/'provenance.rrd'
        LazyChunkStream.merge(reader.stream(),LazyChunkStream.from_iter([Chunk.from_property('calib',rr.AnyValues(calib_exo_refined=refined,calib_exo_gate_json=json.dumps(decision.get('metrics',{}))))])).write_rrd(enriched,application_id=store.application_id,recording_id=store.recording_id)
        enriched.replace(current)
        metrics={'refine_s':time.perf_counter()-started,'calib_exo_refined':refined,'calib_fallback_reason':fallback}
        started=time.perf_counter()
        rate=json.loads((work/'s1/complete.json').read_text())['rate']
        import cache2d
        original=keypoints.detect_keypoints
        original_models=keypoints.keypoint_models
        original_selection=keypoint_geometry.select_wearers
        keypoints.keypoint_models=lambda *args,**kwargs: (None,None)
        cached=cache2d.load(work/'s1/rtmw-cache')
        keypoints.detect_keypoints=lambda *args,**kwargs:cached
        selected_source=work/'s1/rtmw-cache/source.json'
        if selected_source.exists() and json.loads(selected_source.read_text()).get('cache')=='original_s1_selected':
            import numpy as np
            # Restored immutable S1 points already contain the selected wearer.
            # Preserve those exact observations; there are no other candidates to select.
            keypoint_geometry.select_wearers=lambda camera,anchors,boxes,scores:np.zeros(len(boxes),dtype=np.int64)
            metrics['rtmw_input_source']=json.loads(selected_source.read_text())
        try:
            keypoints.main(keypoints.Config(calib_dir=final,output_dir=out/'keypoints',rate=rate,segment_ids=(segment,),catalog_url=server.url()))
        finally:
            keypoints.detect_keypoints=original
            keypoints.keypoint_models=original_models
            keypoint_geometry.select_wearers=original_selection
        metrics['keypoints_s']=time.perf_counter()-started
    layers={'base':work/'s1/align'/f'{segment}.rrd','calib':out/'keypoints'/f'{segment}.rrd','egohand_calib':work/'s1/calib/egohand'/f'{segment}.rrd'}
    if not layers['calib'].exists():layers['calib']=current
    if decision.get('landmarks_path'):
        path=out/'exo_landmarks'/f'{segment}.rrd';write_landmarks(path,segment,from_json(Landmarks,Path(decision['landmarks_path']).read_text()));layers['exo_landmarks']=path
    started=time.perf_counter()
    # Keep the raw layer separate, so S1 evidence is replaced after final wearer selection.
    if (out/'keypoints'/f'{segment}.rrd').exists():
        reader=RrdReader(layers['calib']);store=reader.recordings()[0]
        is_raw=lambda c:str(c.entity_path).endswith('/pinhole/pred/rtmw/coco133_uv')
        raw=out/'rtmw_2d.rrd';clean=out/'calib-published.rrd'
        LazyChunkStream.from_iter(c for c in reader.stream() if is_raw(c)).write_rrd(raw,application_id=store.application_id,recording_id=store.recording_id)
        LazyChunkStream.from_iter(c for c in reader.stream() if not is_raw(c)).write_rrd(clean,application_id=store.application_id,recording_id=store.recording_id)
        layers['calib']=clean;layers['rtmw_2d']=raw
    publish(segment,layers)
    metrics['ship_register_s']=time.perf_counter()-started
    metrics['gpu_peak_mib']=torch.cuda.max_memory_allocated()/2**20
    metrics['host_peak_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (out/'complete.json').write_text(json.dumps(metrics))
    return metrics


def raw_rtmw(segment: str, videos=None) -> dict:
    """Detect once; publish selected raw points while caching every candidate for S2."""
    import cache2d
    import numpy as np
    from dataforge.hands import log_keypoints2d
    from selfcap_curate.keypoints import read_anchors, read_cameras, select_wearers
    from selfcap_curate.keypoints_gpu import detect_keypoints
    work=root(segment)/'s1'; started=time.perf_counter()
    with private(segment) as server:
        dataset=server.client().get_dataset(DATASET_NAME); record=open_segment(dataset,segment);sync=read_time_sync(dataset,segment)
        if videos:
            record=replace(record,videos=videos[segment].videos,hands=videos[segment].hands)
        rate=json.loads((work/'complete.json').read_text())['rate']
        def skipped(reason):
            raw=work/'rtmw_2d.rrd'
            with rr.RecordingStream('dataforge',recording_id=segment,send_properties=False) as rec:
                rec.save(raw);rec.send_property('rtmw_2d',rr.AnyValues(skip_reason=reason));rec.flush()
            publish(segment,{'base':work/'align'/f'{segment}.rrd','sapiens2':work/'align/sapiens2'/f'{segment}.rrd','calib':work/'calib'/f'{segment}.rrd','rtmw_2d':raw})
            return {'rtmw_skipped':reason,'rtmw_detect_publish_s':time.perf_counter()-started}
        with Server(host='127.0.0.1',port=0,datasets={DATASET_NAME:[work/'calib'/f'{segment}.rrd']}) as cs:
            cameras=read_cameras(cs.client().get_dataset(DATASET_NAME),segment)
        if len(cameras)<2:
            return skipped(f'only {len(cameras)} calibrated exo cameras; at least two required')
        cache=work/'rtmw-cache'
        if list(cache.glob('*.parquet')): detections=cache2d.load(cache)
        else:
            try:
                detections=detect_keypoints(record,*keypoint_models(),rate=rate,sync=sync)
            except ValueError as error:
                if 'need two synced videos' not in str(error):raise
                return skipped(str(error))
            if len(detections.times)<2 or not any(view.scores.size and np.any(view.scores>=0.3) for view in detections.views.values()):
                return skipped('no usable RTMW detections in calibrated exo cameras')
            cache2d.save(cache,detections)
        anchors=read_anchors(record,detections.times,sync)
        raw=work/'rtmw_2d.rrd'
        with rr.RecordingStream('dataforge',recording_id=segment,send_properties=False) as rec:
            rec.save(raw)
            for phone,camera in cameras.items():
                if phone not in detections.views:continue
                view=detections.views[phone];people=select_wearers(camera,anchors,view.boxes,view.box_scores)
                xy=np.full((len(people),133,2),np.nan,np.float32);scores=np.zeros((len(people),133),np.float32)
                rows=np.flatnonzero(people>=0);xy[rows]=view.xy[rows,people[rows]];scores[rows]=view.scores[rows,people[rows]]
                log_keypoints2d(rec,times_ns=np.rint(detections.times*1e9).astype(np.int64),frame_indices=None,positions=xy,confidence=scores,path=camera.entity+'/pinhole/pred/rtmw/coco133_uv',color_range=(0.7,1.0))
            rec.flush()
    publish(segment,{'base':work/'align'/f'{segment}.rrd','sapiens2':work/'align/sapiens2'/f'{segment}.rrd','calib':work/'calib'/f'{segment}.rrd','rtmw_2d':raw})
    shutil.rmtree(work/'base',ignore_errors=True)  # Owned fetched copy; the aligned recording is durable.
    return {'rtmw_detect_publish_s':time.perf_counter()-started}


def publish(segment: str, layers: dict) -> None:
    dataset=CatalogClient(CATALOG).get_dataset(name=DATASET)
    for layer,path in layers.items():
        if not path.exists():continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        remote=f'{REMOTE}/{layer}/{segment}-{digest}.rrd'
        subprocess.run(['ssh','spark',f'mkdir -p {REMOTE}/{layer}'],check=True,timeout=60)
        subprocess.run(['scp','-q',str(path),f'spark:{remote}'],check=True,timeout=180)
        dataset.register([f'file://{remote}'],layer_name=layer,on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        manifest=root(segment)/'publish-receipts.json'
        receipts=json.loads(manifest.read_text()) if manifest.exists() else {}
        receipts[layer]={'uri':f'file://{remote}','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'registered_at':time.time()}
        temporary=manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipts,indent=2));temporary.replace(manifest)
