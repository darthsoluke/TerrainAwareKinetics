"""Task allocation helpers; no changes to PARC generation or optimization."""

from pathlib import Path


def select_jobs(plan, role=0, job_ids=None, explicit=False):
    if explicit:
        requested = set(job_ids or [])
        if not requested or not requested.issubset({job['id'] for job in plan['jobs']}):
            raise ValueError('Explicit jobs must name existing scenes')
        return [job for job in plan['jobs'] if job['id'] in requested]
    jobs = [job for index, job in enumerate(plan['jobs']) if (index//6 + index%6) % 2 == role]
    return [job for job in jobs if not job_ids or job['id'] in job_ids]


def ready_jobs(plans, reserved, processed):
    result, seen = [], set()
    for collection, plan in plans:
        for job in plan['jobs']:
            if job['id'] in seen:
                raise ValueError('Duplicate scene identity across campaign plans')
            seen.add(job['id'])
            if (job['id'] not in reserved and job['id'] not in processed and
                    not (collection/job['id']/'receipt.json').exists()):
                result.append((collection, job))
    return result


def idle_cards(csv_text, allowed, assigned):
    result = []
    for line in csv_text.splitlines():
        index, memory, utilization = [part.strip() for part in line.split(',')]
        index = int(index)
        if index in allowed and index not in assigned and float(memory) <= 1024 and float(utilization) <= 5:
            result.append(index)
    return result


def worker_exit_action(returncode, all_receipts_present, adopted=False):
    if returncode == 75 and not adopted and not all_receipts_present:
        return 'retry_when_idle'
    if returncode != 0 or not all_receipts_present:
        return 'failed'
    return 'complete'


def process_identity(pid, proc_root=Path('/proc')):
    root = proc_root/str(pid)
    try:
        command = (root/'cmdline').read_bytes().replace(b'\0', b' ').decode()
        # The command name inside parentheses can contain spaces.
        stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid':pid, 'command':command, 'start_ticks':stat[19], 'state':stat[0]}
    except FileNotFoundError:
        return None


def process_matches(identity, proc_root=Path('/proc')):
    actual = process_identity(identity['pid'], proc_root)
    return (actual is not None and actual['state'] != 'Z' and
            actual['command'] == identity['command'] and actual['start_ticks'] == identity['start_ticks'])
