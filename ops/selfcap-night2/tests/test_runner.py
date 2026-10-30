import json
import sqlite3
from contextlib import contextmanager

import pytest
import runner


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'DB', tmp_path / 'state.sqlite')
    monkeypatch.setattr(runner, 'CONFIG', tmp_path / 'config.json')
    runner.CONFIG.write_text(json.dumps({'limit': 2, 'label_command': ['label'], 'soma_command': ['soma'],
        'max_inflight': 40, 's1_cutoff': 0, 'label_cutoff': 0, 'fallback_enabled': False}))
    with runner.connect() as db:
        db.executescript("create table jobs(segment text primary key,ordinal integer,state text,owner text,updated real,s1_ready real,info text,error text); create table workers(name text primary key, heartbeat real, failures integer); create table events(ts real,segment text,stage text,seconds real,status text,worker text);")
        db.execute("insert into jobs values('test',0,'labelled',null,0,0,'{}',null)")
    return tmp_path


def test_stale_attempt_cannot_update_same_owner_new_claim(queue):
    with runner.connect() as db:
        db.execute("update jobs set state='s2',owner='s2-0',updated=2")
    old_job = {'segment': 'test', 'updated': 1}
    with runner.owned_claim(old_job, 's2', 's2-0') as db:
        assert db is None
    current_job = {'segment': 'test', 'updated': 2}
    with runner.owned_claim(current_job, 's2', 's2-0') as db:
        assert db is not None


@pytest.mark.parametrize('error', [runner.InfrastructureError('broken'), RuntimeError('bad')])
def test_stale_failure_does_not_park_new_owner_or_count_failure(queue, monkeypatch, error):
    claimed = ({'segment': 'test', 'ordinal': 0, 'info': '{}', 's1_ready': 0, 'updated': 1}, 'labelling')
    calls = iter([claimed])
    monkeypatch.setattr(runner, 'claim', lambda *args: next(calls))
    monkeypatch.setattr(runner.threading.Thread, 'start', lambda self: None)
    def fail(*args):
        with runner.connect() as db:
            db.execute("update jobs set state='s2',owner='new-owner',updated=2")
        raise error
    monkeypatch.setattr(runner, 'external', fail)
    with pytest.raises(StopIteration):
        runner.work('label', 0)
    with runner.connect() as db:
        assert tuple(db.execute('select state,owner,updated from jobs').fetchone()) == ('s2', 'new-owner', 2.0)
        assert db.execute('select count(*) from events').fetchone()[0] == 0
        assert db.execute('select failures from workers').fetchone()[0] == 0


def test_config_keeps_last_complete_read_during_partial_legacy_write(queue):
    previous = runner.config()
    runner.CONFIG.write_text('{')
    assert runner.config() == previous
    runner.CONFIG.write_text(json.dumps({**previous, 'limit': 3}))
    assert runner.config()['limit'] == 3


def test_atomic_config_writer_readers_see_only_complete_documents(queue):
    import threading
    errors = []
    stop = threading.Event()
    def read():
        while not stop.is_set():
            try:
                assert json.loads(runner.CONFIG.read_text())['limit'] in (2, 3)
            except Exception as error:
                errors.append(error)
    thread = threading.Thread(target=read)
    thread.start()
    try:
        for value in range(40):
            runner.write_config({'limit': 2 + value % 2, 'payload': 'x' * 10000})
    finally:
        stop.set()
        thread.join()
    assert not errors
