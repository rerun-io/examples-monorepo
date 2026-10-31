"""Publication-gated local cleanup and S1 disk-pressure checks for night2."""
import json
import os
import shutil
import time
from pathlib import Path

MIN_FREE_BYTES = 60 * 1024**3
KEEP_SUFFIXES = {'.json', '.log', '.txt', '.md', '.toml', '.lock'}
_last_disk_log = {}


def storage_available(root: Path, minimum: int = MIN_FREE_BYTES) -> bool:
    free = shutil.disk_usage(root).free
    if free < minimum:
        now = time.monotonic()
        if now - _last_disk_log.get(str(root), -1e9) >= 60:
            print(f'STORAGE_PAUSE_S1 free_bytes={free} required_bytes={minimum}', flush=True)
            _last_disk_log[str(root)] = now
        return False
    if str(root) in _last_disk_log:
        print(f'STORAGE_RESUME_S1 free_bytes={free} required_bytes={minimum}', flush=True)
        del _last_disk_log[str(root)]
    return True


def prune_tree(root: Path, *, dry_run: bool = False) -> dict:
    """Delete heavy files only, without following links or removing the report directory."""
    if root.is_symlink():
        raise ValueError(f'Refusing symlink cleanup root: {root}')
    removed = 0
    size = 0
    if not root.exists():
        return {'files_removed': 0, 'bytes_removed': 0, 'dry_run': dry_run}
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files + directories:
            path = Path(current) / name
            if path.is_symlink():
                if not dry_run:
                    path.unlink()
                removed += 1
            elif path.is_file() and path.suffix.lower() not in KEEP_SUFFIXES:
                removed += 1
                size += path.stat().st_size
                if not dry_run:
                    path.unlink()
            elif path.is_dir() and not dry_run:
                try:
                    path.rmdir()
                except OSError:
                    pass  # Retained reports keep this directory nonempty.
    return {'files_removed': removed, 'bytes_removed': size, 'dry_run': dry_run}


def verify_publication(segment: str, *, manifest: Path | None = None, include_soma: bool = True) -> dict:
    """Read catalog registration metadata and real keypoint/mesh chunks before deletion."""
    from rerun.catalog import CatalogClient
    client = CatalogClient('rerun+http://dgx-spark:51235')
    dataset = client.get_dataset(name='selfcap-post-ep1-v2')
    view = dataset.filter_segments(segment)
    table = view.segment_table().select('rerun_layer_names', 'rerun_storage_urls').to_arrow_table()
    if len(table) != 1:
        raise ValueError(f'{segment}: missing or ambiguous catalog segment')
    layers = table['rerun_layer_names'][0].as_py()
    urls = table['rerun_storage_urls'][0].as_py()
    registered = dict(zip(layers, urls, strict=True))
    required = {'base', 'calib', 'sapiens2', 'rtmw_2d', 'egohand_calib'}
    if include_soma:
        required.add('soma')
    if not required <= set(registered):
        raise ValueError(f'{segment}: missing final layers {required - set(registered)}')
    if manifest is not None:
        receipts = json.loads(manifest.read_text())
        for layer, receipt in receipts.items():
            if not include_soma and layer == 'soma':
                continue
            if registered.get(layer) != receipt['uri']:
                raise ValueError(f'{segment}: {layer} registration differs from successful publication receipt')
    probes = ['/world/pred/rtmw_triangulated/coco133_xyz', '/world/pred/soma_x/mesh', *[f'/world/rig_{rig:02d}/cam_00/pinhole/pred/rtmw/coco133_uv' for rig in range(4)]]
    frame = view.filter_contents(probes).reader(index='video_time')
    names = [n for n in frame.schema().names if n.endswith(('Points3D:positions', 'Points2D:positions', 'Mesh3D:vertex_positions'))]
    data = frame.select(*names).limit(64).to_arrow_table()
    for suffix in (('Points3D:positions', 'Points2D:positions', 'Mesh3D:vertex_positions') if include_soma else ('Points3D:positions', 'Points2D:positions')):
        columns = [n for n in names if n.endswith(suffix)]
        if not any(any(value is not None and len(value) for value in data[n].to_pylist()) for n in columns):
            raise ValueError(f'{segment}: no readable {suffix} in final catalog chunks')
    return {'verified_at': time.time(), 'layers': sorted(registered), 'probe_rows': len(data)}


def complete_cleanup(segment: str, work: Path, *, verify=verify_publication, dry_run: bool = False) -> dict:
    """Verification runs first; failure leaves both local trees untouched."""
    if '/' in segment or not segment.startswith('selfcap__'):
        raise ValueError('Invalid cleanup segment')
    receipt = verify(segment, manifest=work / 'publish-receipts.json')
    runner_files = prune_tree(work, dry_run=dry_run)
    soma_files = prune_tree(Path('/tmp/soma-batch-run/selfcap-post-ep1-v2') / segment, dry_run=dry_run)
    report = {'readback': receipt, 'runner': runner_files, 'soma': soma_files}
    if not dry_run:
        work.mkdir(parents=True, exist_ok=True)
        (work / 'cleanup.json').write_text(json.dumps(report, indent=2))
    print('CLEANUP_DONE', segment, json.dumps(report), flush=True)
    return report


def gpu_available(minimum_mib: int = 16384) -> bool:
    """Gate S1 on actual device-wide free memory, including other processes."""
    import subprocess
    try:
        result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        free=int(result.stdout.strip().splitlines()[0])
    except (OSError,ValueError,subprocess.SubprocessError) as error:
        print('GPU_WAIT_S1 cannot read free memory:',str(error),flush=True)
        return False
    key='gpu0'
    if free<minimum_mib:
        now=time.monotonic()
        if now-_last_disk_log.get(key,-1e9)>=60:
            print(f'GPU_WAIT_S1 free_mib={free} required_mib={minimum_mib}',flush=True)
            _last_disk_log[key]=now
        return False
    if key in _last_disk_log:
        print(f'GPU_RESUME_S1 free_mib={free} required_mib={minimum_mib}',flush=True)
        del _last_disk_log[key]
    return True
