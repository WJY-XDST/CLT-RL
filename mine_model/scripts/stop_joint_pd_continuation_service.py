"""Save an owned checkpoint; distinguish known host shutdown from manual stop."""
import argparse
import json
from pathlib import Path
import subprocess

from auto_train_mine import ROOT, request_checkpoint_stop, write_json


def stop_reason(system_state):
    # An ambiguous service stop is treated as manual; never revoke a pause.
    return 'host_shutdown' if system_state.strip() == 'stopping' else 'service_stop_requested'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if (output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_')
            or not (output/'.controller.lock').is_file()):
        raise ValueError('Stop only an owned cumulative direct-PD controller')
    state=json.loads((output/'status.json').read_text())
    if state.get('state') != 'completed_model_saved' and not (output/'STOP').exists():
        result=subprocess.run(['systemctl','is-system-running'],capture_output=True,text=True,timeout=10)
        write_json(output/'STOP',dict(reason=stop_reason(result.stdout),checkpoint_before_exit=True))
    request_checkpoint_stop(output)
