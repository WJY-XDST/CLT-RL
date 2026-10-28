"""Wait for one owned review with proof-scoped state and process-scoped temp.

The original qualification, source checks, unit observations and review exec
are unchanged. Different proofs in the same directory cannot share state or
rename each other's temporary file. No PPO process is restarted here.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualification',type=Path,required=True)
    cli.add_argument('--review-script',type=Path,required=True)
    args=cli.parse_args()
    proof_path,script=args.qualification.resolve(),args.review_script.resolve()
    proof=json.loads(proof_path.read_text())
    if (not proof['qualified'] or script.parent!=ROOT/'mine_model/scripts'
            or proof['source_pins'].get(str(script))!=sha(script)):
        raise ValueError('Require the exact owned, qualified review entry')
    digest=sha(proof_path)
    unit=proof['after_unit']
    state_path=proof_path.with_name('post200_' + proof_path.stem + '_' + digest[:12] + '_lightweight_wait_state.json')
    while True:
        if sha(proof_path)!=digest or sha(script)!=proof['source_pins'][str(script)]:
            raise ValueError('Qualified review source changed while waiting')
        try:
            raw=subprocess.check_output(['systemctl','--user','show',unit,
                '-p','MainPID','-p','Result','-p','SubState','-p','LoadState'],text=True)
            state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
        except (OSError,subprocess.SubprocessError) as error:
            state=dict(observation_error=repr(error))
        record=dict(observed_at=datetime.now().astimezone().isoformat(),
            after_unit=unit,unit_state=state,training_modified=False,PPO_restarted=False,
            simulator_started=False,policy_accepted=False)
        temp=state_path.with_name(state_path.name + '.' + str(os.getpid()) + '.tmp')
        temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        temp.replace(state_path)
        if state.get('MainPID')=='0' and state.get('Result')=='success' and state.get('SubState')=='exited':
            break
        if state.get('SubState') in ('failed','dead') or state.get('LoadState')=='not-found':
            raise RuntimeError('Inspect the actual training unit; this waiter never restarts PPO: '+repr(state))
        time.sleep(15.)
    os.execv(str(ROOT/'run_python.sh'),[str(ROOT/'run_python.sh'),str(script)])


if __name__=='__main__':
    main()
