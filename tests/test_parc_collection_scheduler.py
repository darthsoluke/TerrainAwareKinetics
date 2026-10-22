"""Scaling must preserve active ownership, source identities and busy GPUs."""

from pathlib import Path

import pytest

from scripts.parc_collection_scheduler import (
    idle_cards, process_identity, process_matches, ready_jobs, select_jobs, worker_exit_action,
)


def plan(count=24, offset=0):
    return {'jobs':[{'id':f'scene_{i}'} for i in range(offset,offset+count)]}


def test_legacy_workers_keep_exact_original_disjoint_assignments():
    source=plan()
    a,b=({j['id'] for j in select_jobs(source,role)} for role in (0,1))
    assert len(a)==len(b)==12
    assert not a & b
    assert a | b == {j['id'] for j in source['jobs']}


def test_explicit_dispatch_does_not_filter_by_legacy_role():
    source=plan()
    assert select_jobs(source,0,['scene_1'])==[]
    assert select_jobs(source,0,['scene_1'],explicit=True)==[{'id':'scene_1'}]
    with pytest.raises(ValueError):select_jobs(source,job_ids=['unknown'],explicit=True)
    with pytest.raises(ValueError):select_jobs(source,explicit=True)


def test_busy_user_card_and_owned_cards_are_excluded_even_at_zero_utilization():
    rows='0,12949,0\n1,17,0\n2,900,0\n3,100,75\n4,18,0\n5,57,0\n6,55,0\n7,55,0'
    assert idle_cards(rows,range(8),{2,3})==[1,4,5,6,7]
    assert idle_cards(rows,[4,6],{})==[4,6]


def test_existing_worker_queues_are_reserved_while_free_cards_take_new_wave(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new'
    old_plan,new_plan=plan(),plan(offset=24)
    reserved={j['id'] for j in old_plan['jobs']}
    queue=ready_jobs([(old,old_plan),(new,new_plan)],reserved,set())
    assert len(queue)==24
    assert all(directory==new for directory,_ in queue)
    issued=queue[:5]
    assert len({job['id'] for _,job in issued})==5
    reserved.update(job['id'] for _,job in issued)
    assert len(ready_jobs([(old,old_plan),(new,new_plan)],reserved,set()))==19


def test_completed_and_admitted_jobs_are_not_dispatched_again(tmp_path):
    folder=tmp_path/'scene_1';folder.mkdir();(folder/'receipt.json').write_text('{}')
    queue=ready_jobs([(tmp_path,plan(4))],{'scene_0'},{'scene_2'})
    assert [job['id'] for _,job in queue]==['scene_3']


def test_duplicate_scene_identity_is_an_error(tmp_path):
    with pytest.raises(ValueError,match='Duplicate'):
        ready_jobs([(tmp_path/'a',plan()),(tmp_path/'b',plan())],set(),set())


def test_pid_reuse_or_zombie_is_not_a_running_owned_worker(tmp_path):
    process=tmp_path/'123';process.mkdir()
    (process/'cmdline').write_bytes(b'python\0worker.py\0')
    fields=['S']+['0']*18+['123456']
    (process/'stat').write_text('123 (python worker) '+' '.join(fields))
    identity=process_identity(123,tmp_path)
    assert identity['start_ticks']=='123456'
    assert process_matches(identity,tmp_path)
    fields[19]='999999'
    (process/'stat').write_text('123 (python worker) '+' '.join(fields))
    assert not process_matches(identity,tmp_path)
    fields[19]='123456';fields[0]='Z'
    (process/'stat').write_text('123 (python worker) '+' '.join(fields))
    assert not process_matches(identity,tmp_path)
    assert process_identity(999,tmp_path) is None


def test_gpu_claim_race_requeues_but_execution_failures_do_not_disappear():
    assert worker_exit_action(75,False)=='retry_when_idle'
    assert worker_exit_action(1,False)=='failed'
    assert worker_exit_action(0,False)=='failed'
    assert worker_exit_action(75,False,adopted=True)=='failed'
    assert worker_exit_action(0,True)=='complete'
