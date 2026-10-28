"""Read-only cumulative episode-event monitor; never controls training.

Parse complete RSL iteration blocks so an unsafe event remains visible after
later iterations report zero. Episode_Count uses the project's event-weighted
rollup, not repeated per-step averages. Reason counts are recovered from the
four-decimal printed ratios and can overlap across reasons.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'mine_model/results'


def summarize(directory):
    directory = Path(directory).resolve()
    text = (directory/'training.log').read_text(errors='replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    headers = list(re.finditer(r'Learning iteration\s+(\d+)/(\d+)',text))
    rows, events, seen = [], [], set()
    for index,header in enumerate(headers):
        block = text[header.start():headers[index+1].start() if index+1<len(headers) else len(text)]
        if 'Total timesteps:' not in block or 'ETA:' not in block:
            continue  # A partially written log block is not a completed record.
        iteration = int(header[1])
        if iteration in seen:
            raise ValueError('Duplicate complete printed iteration; audit resumed logging before counting')
        seen.add(iteration)
        values = {key:float(value) for key,value in re.findall(
            r'(Episode_(?:Count|Termination)/[^:\s]+):\s*([-+0-9.eE]+)',block)}
        ended = int(round(values.get('Episode_Count/all_ended',0.)))
        unsafe = int(round(values.get('Episode_Count/all_unsafe',0.)))
        numerical = int(round(values.get('Episode_Count/numerical',0.)))
        if min(ended,unsafe,numerical)<0 or unsafe>ended:
            raise ValueError('Invalid episode-event counts')
        near = int(round(values.get('Episode_Termination/unsafe_within_1s_of_command',0.)*unsafe))
        row = dict(log_iteration=iteration,ended=ended,unsafe=unsafe,numerical=numerical,
            unsafe_within_1s_of_command=near)
        rows.append(row)
        if unsafe or numerical:
            reasons = {key.split('/',1)[1]:int(round(values.get(key,0.)*ended)) for key in (
                'Episode_Termination/base_contact','Episode_Termination/pitch',
                'Episode_Termination/roll','Episode_Termination/fallen',
                'Episode_Termination/leg_contact','Episode_Termination/low_height',
                'Episode_Termination/kinematic_singularity')}
            events.append(dict(**row,reason_counts_from_printed_ratios=reasons,
                high_height_unsafe=int(round(values.get('Episode_Count/high_height_unsafe',0.)))))
    def total(selected):
        ended = sum(r['ended'] for r in selected)
        unsafe = sum(r['unsafe'] for r in selected)
        return dict(ended=ended,unsafe=unsafe,numerical=sum(r['numerical'] for r in selected),
            unsafe_rate=unsafe/ended if ended else None,
            unsafe_within_1s_of_command=sum(r['unsafe_within_1s_of_command'] for r in selected))
    return dict(updated_at=datetime.now().astimezone().isoformat(),directory=str(directory),
        complete_iterations=len(rows),last_log_iteration=rows[-1]['log_iteration'] if rows else None,
        cumulative=total(rows),last20=total(rows[-20:]),events=events,
        history_not_cleared_by_later_zero=True,training_modified=False,policy_accepted=False,
        limitation='All4096 episode counts from complete printed rollout summaries. No per-event '
        'environment IDs or physical-state traces; rounded reason counts can overlap. '
        'Correlation with command changes is not proof of a sampler or controller bug.')


def save(directory):
    report=summarize(directory)
    for path in (Path(directory)/'training_safety_history.json',RESULTS/'joint_pd_training_event_history.json'):
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    return report


def main():
    previous=None
    while True:
        try:
            pointer=json.loads((RESULTS/'joint_pd_active_joint_tracking.json').read_text())
            directory=Path(pointer['directory'])
            if directory.parent.resolve()!=RESULTS.resolve():
                raise ValueError('Only monitor owned training outputs')
            report=save(directory)
            key=(str(directory),report['cumulative']['unsafe'],report['cumulative']['numerical'])
            if key!=previous:
                print(json.dumps(dict(directory=str(directory),last=report['last_log_iteration'],
                    cumulative=report['cumulative'],last20=report['last20'],training_modified=False)),flush=True)
                previous=key
        except (FileNotFoundError,json.JSONDecodeError):
            pass
        except Exception as error:
            (RESULTS/'joint_pd_event_history_error.json').write_text(json.dumps(dict(
                updated_at=datetime.now().astimezone().isoformat(),error=repr(error),training_modified=False))+'\n')
            raise
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',type=Path)
    args=parser.parse_args()
    if args.once:
        r=save(args.once)
        print(json.dumps({k:v for k,v in r.items() if k!='events'},indent=2))
    else:
        main()
