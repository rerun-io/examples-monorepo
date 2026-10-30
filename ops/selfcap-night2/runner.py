"""Durable SelfCap S1 -> labels -> S2 -> SOMA queues. All writes target ep1-v2."""
import argparse
import contextlib
import datetime as dt
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from batch_reporting import report, stage_paused, stage_result
from infrastructure import is_infrastructure_error
from repeated_errors import repeated_stage_error, is_soma_input_limitation

class InfrastructureError(RuntimeError):
    pass

class InputLimitation(RuntimeError):
    pass


ROOT=Path('/tmp/selfcap-night2');DB=ROOT/'state.sqlite';CONFIG=ROOT/'config.json'
PARENT='773be1d3-a483-4a1c-9fcf-6c76b243dcec'
TZ=ZoneInfo('America/Chicago')


_last_config = {}


def config():
    """Keep workers alive across transient legacy config truncation."""
    for attempt in range(3):
        try:
            value=json.loads(CONFIG.read_text())
            if not isinstance(value,dict):raise ValueError('config must be a JSON object')
            _last_config[CONFIG]=value
            return value.copy()
        except (OSError,json.JSONDecodeError,ValueError):
            if attempt<2:time.sleep(0.02)
    if CONFIG in _last_config:
        print('CONFIG_READ_RETRY using last complete configuration',flush=True)
        return _last_config[CONFIG].copy()
    raise ValueError(f'{CONFIG}: no complete configuration available')


def write_config(value):
    """Publish a complete document. Operator writers must use this protocol too."""
    with tempfile.NamedTemporaryFile(mode='w',dir=CONFIG.parent,prefix=f'.{CONFIG.name}-',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            stream.write(json.dumps(value,indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(CONFIG)
    finally:
        temporary.unlink(missing_ok=True)

def connect():
    c=sqlite3.connect(DB,timeout=30);c.row_factory=sqlite3.Row;c.execute('pragma journal_mode=WAL');return c


def notify(message):
    """Observability must never change a job's processing result."""
    try:
        result=subprocess.run(['/home/pablo/.pixi/envs/paseo/bin/paseo','send',PARENT,'--behavior','steer','--no-wait',message],timeout=30,check=False,capture_output=True,text=True,env={**os.environ,'PATH':'/home/pablo/.pixi/envs/paseo/bin:'+os.environ.get('PATH','')})
        if result.returncode:
            print('NOTIFY_FAILED',result.returncode,result.stderr[-500:],flush=True)
    except Exception as error:
        print('NOTIFY_FAILED',type(error).__name__,str(error),flush=True)


def initialize():
    ROOT.mkdir(exist_ok=True)
    if not CONFIG.exists():
        today=dt.datetime.now(TZ).date()
        write_config({'limit':1,'label_command':None,'soma_command':None,'label_workers':2,'soma_workers':2,'max_inflight':40,'label_cutoff':dt.datetime.combine(today,dt.time(7),TZ).timestamp(),'s1_cutoff':dt.datetime.combine(today,dt.time(7,30),TZ).timestamp()})
    priorities=[line.split('\\t')[0].split()[0] for line in Path('/tmp/selfcap-5090/validation-10.tsv').read_text().splitlines() if line.strip()]
    queue=list(dict.fromkeys(priorities+Path('/tmp/selfcap-night/queue.txt').read_text().split()))
    with connect() as c:
        c.executescript('create table if not exists jobs(segment text primary key,ordinal integer,state text,owner text,updated real,s1_ready real,info text,error text);create table if not exists events(ts real,segment text,stage text,seconds real,status text,worker text);create table if not exists workers(name text primary key,heartbeat real,failures integer default 0);')
        for n,seg in enumerate(queue):c.execute("insert or ignore into jobs values(?,?,'queued',null,?,null,'{}',null)",(seg,n,time.time()))
    print(f'{len(queue)} queued; enabled limit={config()["limit"]}',flush=True)


def claim(role,name):
    cfg=config();now=time.time()
    if cfg.get("corpus_paused",False):return None
    with connect() as c:
        c.execute('begin immediate')
        c.execute("update jobs set state='calib_input_limited',owner=null,error=null where ordinal<? and state in ('s1_done','labelled','label_timeout') and json_extract(info,'$.s1.rtmw_skipped') is not null",(cfg['limit'],))
        if cfg.get('fallback_enabled',True):
            c.execute("update jobs set state='label_timeout',owner=null,updated=? where ordinal<? and state in ('s1_done','labelling') and (ordinal>0 or ?) and (s1_ready<? or (owner is null and ? >= ?))",(now,cfg['limit'],bool(cfg['label_command']),now-1200,now,cfg['label_cutoff']))
        job=None;stage=role
        if role=='gpu':
            job=None if cfg.get('s2_paused',False) or cfg.get('s2_separate',False) or stage_paused(c,'s2') else c.execute("select * from jobs where ordinal<? and state in ('labelled','label_timeout') order by ordinal limit 1",(cfg['limit'],)).fetchone();stage='s2'
            pending=c.execute("select count(*) from jobs where ordinal<? and state not in ('queued','done','failed','calib_input_limited')",(cfg['limit'],)).fetchone()[0]
            if job is None and now<cfg['s1_cutoff'] and pending<cfg['max_inflight'] and not stage_paused(c,'s1'):
                from storage import storage_available, gpu_available
                if not storage_available(ROOT,cfg.get('minimum_free_bytes',60*1024**3)):return None
                if not gpu_available(cfg.get('minimum_free_gpu_mib',16384)):return None
                job=c.execute("select * from jobs where ordinal<? and state='queued' order by ordinal limit 1",(cfg['limit'],)).fetchone();stage='s1'
        elif role=='s2' and not cfg.get('s2_paused',False) and not stage_paused(c,'s2'):
            job=c.execute("select * from jobs where ordinal<? and state in ('labelled','label_timeout') order by ordinal limit 1",(cfg['limit'],)).fetchone();stage='s2'
        elif role=='label' and cfg['label_command'] and not cfg.get('labels_paused',False) and now<cfg['label_cutoff'] and not stage_paused(c,'labelling'):
            job=c.execute("select * from jobs where ordinal<? and state='s1_done' and owner is null order by ordinal limit 1",(cfg['limit'],)).fetchone();stage='labelling'
        elif role=='soma' and cfg['soma_command'] and not cfg.get('soma_paused',False) and not stage_paused(c,'soma'):
            job=c.execute("select * from jobs where ordinal<? and state='soma' and owner is null order by ordinal limit 1",(cfg['limit'],)).fetchone()
        if job is None:return None
        c.execute('update jobs set state=?,owner=?,updated=? where segment=?',(stage,name,now,job['segment']))
        return {**dict(job), 'updated': now},stage


@contextlib.contextmanager
def owned_claim(job, stage, name):
    """Serialize result writes only while this exact claim still owns the job.

    The claim timestamp distinguishes restarts reusing the same worker name.
    This fences database writes, not external artifact publication.
    """
    with connect() as c:
        c.execute('begin immediate')
        current=c.execute('select 1 from jobs where segment=? and state=? and owner=? and updated=?',
                          (job['segment'],stage,name,job['updated'])).fetchone()
        yield c if current else None


def heartbeat(name,stop):
    while not stop.wait(5):
        with connect() as c:c.execute('update workers set heartbeat=? where name=?',(time.time(),name))


def external(command,job,stage,timeout):
    seg=job['segment'];work=ROOT/'runs'/seg;out=work/stage;out.mkdir(parents=True,exist_ok=True)
    initial=next((work/'s1/calib').glob('*.json'))
    fields={'segment':seg,'work':str(work),'s1_dir':str(work/'s1'),'s2_dir':str(work/'s2'),'calib_record':str(initial),'output_dir':str(out),'dataset':'selfcap-post-ep1-v2','catalog_url':'rerun+http://dgx-spark:51235'}
    args=[part.format(**fields) for part in command]
    with (out/'command.log').open('a') as log:
        log_start=log.tell()
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path(config()['pipeline_worktree']))
        try:
            deadline=time.monotonic()+timeout
            while proc.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError(f'{stage} exceeded {timeout}s')
                if stage=='label':
                    with connect() as c:state=c.execute('select state from jobs where segment=?',(seg,)).fetchone()[0]
                    if state!='labelling':raise TimeoutError('label result arrived after S2 fallback')
                time.sleep(1)
            if proc.returncode:
                with (out/'command.log').open() as failed_log:
                    failed_log.seek(log_start)
                    failure_text=failed_log.read()
                if stage=='soma' and is_soma_input_limitation(proc.returncode,failure_text):
                    raise InputLimitation(next(line for line in failure_text.splitlines() if line.startswith('SOMA_INPUT_LIMITATION ')))
                if is_infrastructure_error(failure_text):
                    raise InfrastructureError(f'{stage} code/environment failure: {failure_text[-2500:]}')
                failure_message=next((line.strip() for line in reversed(failure_text.splitlines()) if line.strip()),f'exit {proc.returncode}')
                raise RuntimeError(failure_message)
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    return out


def work(role,index):
    name=f'{role}-{index}'
    with connect() as c:
        c.execute('insert into workers values(?,?,0) on conflict(name) do update set heartbeat=excluded.heartbeat',(name,time.time()))
        for stale in c.execute('select segment,state,updated from jobs where owner=?',(name,)).fetchall():
            state={'s1':'queued','s2':'labelled','labelling':'s1_done'}.get(stale['state'],stale['state'])
            c.execute('update jobs set state=?,owner=null,updated=? where segment=? and state=? and owner=? and updated=?',(state,time.time(),stale['segment'],stale['state'],name,stale['updated']))
        if c.execute('select failures from workers where name=?',(name,)).fetchone()[0]>=3:return
    stop=threading.Event();threading.Thread(target=heartbeat,args=(name,stop),daemon=True).start()
    while True:
        selected=claim(role,name)
        if selected is None:time.sleep(2);continue
        job,stage=selected;seg=job['segment'];info=json.loads(job['info']);started=time.perf_counter();now=time.time()
        try:
            if role in ('gpu','s2'):
                import stages
                import torch
                if role=='gpu':torch.cuda.reset_peak_memory_stats()
                result=stages.s1(seg) if stage=='s1' else stages.s2(seg,info.get('decision'))
                info[stage]=result
                if stage=='s2':
                    from storage import verify_publication
                    info['s2_readback']=verify_publication(seg,manifest=ROOT/'runs'/seg/'publish-receipts.json',include_soma=False)
                if role=='gpu':stages.release()
                target=('calib_input_limited' if result.get('rtmw_skipped') else 's1_done') if stage=='s1' else 'soma'
            elif role=='label':
                try:
                    out=external(config()['label_command'],job,'label',370 if not config().get('fallback_enabled',True) else min(370,max(1,1200-(time.time()-job['s1_ready']))))
                    decision=json.loads((out/'result.json').read_text())
                except InfrastructureError:
                    raise
                except (TimeoutError,RuntimeError) as error:decision={'decision':'timeout' if isinstance(error,TimeoutError) else 'failed','reason':str(error)}
                info['decision']={'refined':decision['decision']=='accepted','metrics':decision.get('gate'), 'decision':decision['decision'],'reason':decision.get('reason'), 'calib_path':str(ROOT/'runs'/seg/'label/calib.json')}
                labels=ROOT/'runs'/seg/'label/landmarks.json'
                if labels.exists():info['decision']['landmarks_path']=str(labels)
                target='calib_input_limited' if decision['decision']=='skipped' else ('labelled' if decision['decision'] in ('accepted','rejected') else 'label_timeout')
            else:
                from storage import verify_publication
                info['s2_readback']=verify_publication(seg,manifest=ROOT/'runs'/seg/'publish-receipts.json',include_soma=False)
                external(config()['soma_command'],job,'soma',1200)
                from storage import verify_publication
                info['final_readback']=verify_publication(seg,manifest=ROOT/'runs'/seg/'publish-receipts.json')
                target='done'
            elapsed=time.perf_counter()-started;info[stage+'_wall_s']=elapsed
            with owned_claim(job,stage,name) as c:
                if c is None:continue
                c.execute('update jobs set state=?,owner=null,updated=?,s1_ready=coalesce(s1_ready,?),info=?,error=null where segment=?',(target,time.time(),time.time() if stage=='s1' else None,json.dumps(info),seg))
                label_ok=role!='label' or info.get('decision',{}).get('decision') in ('accepted','rejected','skipped')
                newly_paused=stage_result(c,stage,label_ok,info.get('decision',{}).get('reason') if not label_ok else None)
                c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'ok' if label_ok else 'failed',name));c.execute('update workers set failures=0 where name=?',(name,))
            if newly_paused:notify(f'STAGE PAUSED after 3 consecutive failures: {stage}. '+json.dumps(report_snapshot()))
            print(name,seg,stage,'ok',round(elapsed,2),flush=True)
            if target=='done':
                if job['ordinal']==0:notify('d31 real batch example DONE: rerun+http://dgx-spark:51235/dataset/18E339F97894103D2f94545c87907d98?segment_id='+seg+' Decision and stage times: '+json.dumps({k:v for k,v in info.items() if k.endswith('_wall_s')})+' decision='+str(info.get('decision',{}).get('decision')))
                if config().get('cleanup_after_success',True):cleanup_finished(seg,info)
        except Exception as error:
            detail=traceback.format_exc();elapsed=time.perf_counter()-started
            if isinstance(error, InfrastructureError) or is_infrastructure_error(detail):
                info['infrastructure_stage']=stage
                info['infrastructure_reason']=str(error)
                with owned_claim(job,stage,name) as c:
                    if c is None:continue
                    c.execute("update jobs set state='infrastructure_waiting',owner=null,error=null,updated=?,info=? where segment=?",(time.time(),json.dumps(info),seg))
                    c.execute("insert into stage_health(stage,failures,paused,reason) values(?,0,1,?) on conflict(stage) do update set failures=0,paused=1,reason=excluded.reason",(stage,str(error)))
                    c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'infrastructure_pause',name))
                notify(f'STAGE PAUSED for code/environment failure, not data failure: {stage} {seg}: {str(error)[-1800:]}')
                continue
            if role=='gpu' and ('OutOfMemoryError' in detail or 'CUDA out of memory' in detail):
                info['oom_requeued_at']=time.time()
                with owned_claim(job,stage,name) as c:
                    if c is None:continue
                    c.execute("update jobs set state='queued',owner=null,error=null,updated=?,info=? where segment=?",(time.time(),json.dumps(info),seg))
                    c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'resource_retry',name))
                notify(f'S1 OOM resource retry, not data failure: {seg}; waiting for16GiB free before another claim')
                stages.release(True)
                sys.exit(1)
            if stage=='soma' and any(reason in str(error) for reason in ('CSV has no orientations', 'need at least two Quest hand and head samples')):
                info.pop('failed_stage',None);info.pop('retries',None)
                info['waiting_reason']='SOMA Quest orientation or head-path reader requires pending fix'
                with owned_claim(job,stage,name) as c:
                    if c is None:continue
                    c.execute("update jobs set state='soma_waiting_fix',owner=null,error=null,updated=?,info=? where segment=?",(time.time(),json.dumps(info),seg))
                    c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'waiting_fix',name))
                print('SOMA_WAITING_FIX',seg,flush=True)
                continue
            if stage in ('s1','soma','s2') and (isinstance(error,InputLimitation) or repeated_stage_error(info,stage,str(error))):
                info.pop('failed_stage',None);info.pop('retries',None)
                info['waiting_reason']=str(error)
                with owned_claim(job,stage,name) as c:
                    if c is None:continue
                    c.execute("update jobs set state=?,owner=null,error=null,updated=?,info=? where segment=?",({'soma':'soma_input_limited','s2':'s2_waiting_fix','s1':'s1_waiting_fix'}[stage],time.time(),json.dumps(info),seg))
                    c.execute("update events set status='input_limited' where segment=? and stage=? and status='failed'",(seg,stage))
                    c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'input_limited',name))
                notify(f'{stage} input limitation or repeated error, parked: {seg}: {error}')
                continue
            with owned_claim(job,stage,name) as c:
                if c is None:continue
                c.execute("update jobs set state='failed',owner=null,updated=?,info=?,error=? where segment=?",(time.time(),json.dumps({**info,'failed_stage':stage}),detail,seg))
                newly_paused=stage_result(c,stage,False,str(error))
                c.execute('insert into events values(?,?,?,?,?,?)',(time.time(),seg,stage,elapsed,'failed',name));c.execute('update workers set failures=failures+1 where name=?',(name,));failures=c.execute('select failures from workers where name=?',(name,)).fetchone()[0]
            print(name,seg,stage,detail,flush=True)
            if newly_paused:notify(f'STAGE PAUSED after 3 consecutive failures: {stage}. '+json.dumps(report_snapshot()))
            if role=='gpu':
                try:stages.release(True)
                except Exception:pass
            notify(f'night2 {name} failed {stage} {seg}: {str(error)[:500]}')
            if failures>=3:return
            if role=='gpu':sys.exit(1) # Restart the process after failure: fresh CUDA context and models.


def cleanup_finished(segment,info):
    from storage import complete_cleanup
    info['cleanup_attempted_at']=time.time()
    try:
        info['cleanup']=complete_cleanup(segment,ROOT/'runs'/segment)
        info.pop('cleanup_pending',None)
    except Exception as error:
        info['cleanup_pending']=str(error)
        print('CLEANUP_PENDING',segment,str(error),flush=True)
    with connect() as c:
        c.execute("update jobs set info=? where segment=? and state='done'",(json.dumps(info),segment))


def report_snapshot():
    with connect() as db:return report(db,config())


def status():
    print(json.dumps(report_snapshot(),indent=2))


def controller():
    while True:
        with connect() as c:
            stale=[r['name'] for r in c.execute('select * from workers where heartbeat<?',(time.time()-120,))]
            for owner in stale:
                for job in c.execute('select * from jobs where owner=?',(owner,)).fetchall():
                    state={'s1':'queued','s2':'labelled','labelling':'s1_done'}.get(job['state'],job['state'])
                    c.execute('update jobs set state=?,owner=null,updated=? where segment=? and state=? and owner=? and updated=?',(state,time.time(),job['segment'],job['state'],owner,job['updated']))
            # Retry an interrupted session twice, retaining its error history and worker failure guard.
            for job in c.execute("select * from jobs where ordinal<? and state='failed' and updated<?",(config()['limit'],time.time()-10)).fetchall():
                info=json.loads(job['info']); retries=info.get('retries',0)
                if retries>=2:continue
                stage=info.get('failed_stage','s1'); info['retries']=retries+1
                state={'s1':'queued','s2':'labelled','soma':'soma','labelling':'s1_done'}[stage]
                c.execute('update jobs set state=?,owner=null,updated=?,info=? where segment=? and state=? and owner is ? and updated=?',(state,time.time(),json.dumps(info),job['segment'],job['state'],job['owner'],job['updated']))
        cfg=config()
        if cfg.get('batch_started_at'):
            snapshot=report_snapshot()
            done=snapshot['terminal']
            marker=ROOT/'batch100-milestones.json'
            sent=json.loads(marker.read_text()) if marker.exists() else []
            for milestone in (10,50,100):
                if done>=milestone and milestone not in sent and (milestone<100 or not cfg.get('redecision_pending',False)):
                    notify(f'BATCH100 {milestone} terminal sessions: '+json.dumps(snapshot))
                    sent.append(milestone);marker.write_text(json.dumps(sent))
            (ROOT/'batch100-status.json').write_text(json.dumps(snapshot,indent=2))
            if done==cfg['limit'] and not cfg.get('redecision_pending',False) and not any(json.loads(r['info']).get('cleanup_pending') for r in connect().execute("select info from jobs where ordinal<?",(cfg['limit'],))):
                for role,count in [('gpu',2),('label',3),('s2',3),('soma',6)]:
                    for index in range(count):subprocess.run(['systemctl','--user','stop',f'selfcap-night2-{role}-{index}'],check=False)
                finished=config();finished['full_go']=False;finished['corpus_paused']=True;write_config(finished)
                return
        if cfg.get('full_go'):
            with connect() as c:
                completed=c.execute("select segment,info from jobs where ordinal<? and state='done'",(cfg['limit'],)).fetchall()
            for completed_job in completed:
                completed_info=json.loads(completed_job['info'])
                if completed_info.get('cleanup_pending') and time.time()-completed_info.get('cleanup_attempted_at',0)>60:
                    cleanup_finished(completed_job['segment'],completed_info)
            with connect() as c:
                pending=c.execute("select count(*) from jobs where ordinal<? and state in ('s1','s1_done','labelling','labelled','label_timeout','s2')",(cfg['limit'],)).fetchone()[0]
                queued=c.execute("select count(*) from jobs where ordinal<? and state='queued'",(cfg['limit'],)).fetchone()[0]
                active_s1=c.execute("select count(*) from jobs where ordinal<? and state='s1'",(cfg['limit'],)).fetchone()[0]
            if active_s1==0 and (queued==0 or time.time()>=cfg['s1_cutoff']):
                for i in range(cfg.get('full_gpu_workers',2)):
                    subprocess.run(['systemctl','--user','stop',f'selfcap-night2-gpu-{i}'],check=False)
                if not stage_paused(c,'soma'):
                    launch('soma',cfg.get('drain_soma_workers',6))
                    current=config();current['active_soma_workers']=cfg.get('drain_soma_workers',6)
                    write_config(current)
        time.sleep(10)


def launch(role,count):
    for i in range(count):
        name=f'selfcap-night2-{role}-{i}'
        if subprocess.run(['systemctl','--user','is-active','--quiet',name]).returncode==0:continue
        loaded=subprocess.run(['systemctl','--user','show','--value','-p','LoadState',name],capture_output=True,text=True).stdout.strip()
        if loaded=='loaded':
            subprocess.run(['systemctl','--user','start',name],check=True)
            continue
        cfg=config()
        args=['systemd-run','--user',f'--unit={name}','--collect',f"--property=WorkingDirectory={cfg['pipeline_worktree']}",'--property=Restart=on-failure','--property=RestartSec=10','--setenv=CUDA_MPS_PIPE_DIRECTORY=/tmp/selfcap-mps/pipe','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONUNBUFFERED=1','--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',f"--setenv=PYTHONPATH={cfg['pipeline_pythonpath']}",f"--setenv=TRITON_PTXAS_BLACKWELL_PATH={Path(cfg['pipeline_worktree'])/'.pixi/envs/selfcap-curate/bin/ptxas'}"]
        if role=='s2':args.append('--setenv=CUDA_VISIBLE_DEVICES=')
        args += [cfg['pipeline_python'],str(Path(cfg['pipeline_worktree'])/'ops/selfcap-night2/runner.py'), 'controller' if role=='controller' else 'worker']
        if role!='controller':args +=[role,str(i)]
        subprocess.run(args,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['setup','worker','controller','status','launch']);parser.add_argument('role',nargs='?');parser.add_argument('index',nargs='?',type=int,default=0);a=parser.parse_args()
    if a.command=='setup':initialize()
    elif a.command=='status':status()
    elif a.command=='worker':work(a.role,a.index)
    elif a.command=='controller':controller()
    elif a.command=='launch':launch(a.role,a.index)
