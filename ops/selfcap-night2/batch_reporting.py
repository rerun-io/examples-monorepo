"""Selected-batch reports and shared stage failure guard."""
import collections
import json
import time


def ensure_health(db):
    db.execute('create table if not exists stage_health(stage text primary key, failures integer not null default 0, paused integer not null default 0, reason text)')


def stage_paused(db, stage):
    ensure_health(db)
    row=db.execute('select paused from stage_health where stage=?',(stage,)).fetchone()
    return bool(row and row[0])


def stage_result(db, stage, ok, reason=None):
    ensure_health(db)
    db.execute('insert or ignore into stage_health(stage,failures,paused) values(?,0,0)',(stage,))
    row=db.execute('select failures,paused from stage_health where stage=?',(stage,)).fetchone()
    failures=0 if ok else row[0]+1
    newly_paused=not row[1] and failures>=3
    db.execute('update stage_health set failures=?,paused=?,reason=? where stage=?',(failures,int(bool(row[1]) or newly_paused),None if ok else reason,stage))
    return newly_paused


def report(db,cfg):
    rows=db.execute('select * from jobs where ordinal<?',(cfg['limit'],)).fetchall()
    counts=dict(collections.Counter(row['state'] for row in rows))
    infos=[json.loads(row['info']) for row in rows]
    decisions=collections.Counter(info.get('decision',{}).get('decision','pending') for info in infos)
    reasons={}
    for decision in ('rejected','timeout','failed'):
        reasons[decision]=dict(collections.Counter(info.get('decision',{}).get('reason','unspecified') for info in infos if info.get('decision',{}).get('decision')==decision))
    job_errors=dict(collections.Counter((json.loads(row['info']).get('failed_stage','unknown')+': '+(row['error'] or 'unspecified').splitlines()[-1]) for row in rows if row['state']=='failed'))
    means={}
    for stage,default in [('s1',25),('labelling',60),('s2',20),('soma',100)]:
        samples=[info[stage+'_wall_s'] for info in infos if isinstance(info.get(stage+'_wall_s'),(int,float))]
        means[stage]=round(sum(samples)/len(samples),2) if samples else default
    remaining={
        's1':sum(row['state'] in ('queued','s1') for row in rows),
        'labelling':sum(row['state'] in ('queued','s1','s1_done','labelling') for row in rows),
        's2':sum(row['state'] in ('queued','s1','s1_done','labelling','labelled','label_timeout','s2') for row in rows),
        'soma':sum(row['state'] not in ('done','failed','calib_input_limited') for row in rows),
    }
    pools={'s1':cfg.get('full_gpu_workers',2),'labelling':cfg.get('full_label_workers',3),'s2':cfg.get('full_s2_workers',3),'soma':cfg.get('active_soma_workers',4)}
    eta=max((remaining[stage]*means[stage]/pools[stage] for stage in remaining),default=0)
    if remaining['soma']:eta+=means['s2']+means['soma']
    ensure_health(db)
    health={row['stage']:{'consecutive_failures':row['failures'],'paused':bool(row['paused']),'reason':row['reason']} for row in db.execute('select * from stage_health')}
    terminal=counts.get('done',0)+counts.get('calib_input_limited',0)+sum(row['state']=='failed' and json.loads(row['info']).get('retries',0)>=2 for row in rows)
    return {'terminal':terminal,'validation_states':dict(collections.Counter(row['state'] for row in rows if row['ordinal']<10)),'selected':len(rows),'states':counts,'refined':decisions['accepted'],'rejected':decisions['rejected'],'timeout':decisions['timeout'],'label_failed':decisions['failed'],'failed_jobs':counts.get('failed',0),'decision_reasons':reasons,'job_errors':job_errors,'mean_stage_seconds':means,'eta_minutes':round(eta/60,1),'stage_health':health,'elapsed_minutes':round((time.time()-cfg.get('batch_started_at',time.time()))/60,1)}
