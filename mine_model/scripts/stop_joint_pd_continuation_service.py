"""Save an owned checkpoint; distinguish known host shutdown from manual stop."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from auto_train_mine import ROOT, request_checkpoint_stop, write_json


def stop_reason(system_state):
    # An ambiguous service stop is treated as manual; never revoke a pause.
    return 'host_shutdown' if system_state.strip() == 'stopping' else 'service_stop_requested'


def main_process_alive(main_pid):
    """ExecStop also runs after natural exit, when systemd unsets MAINPID."""
    if not main_pid:
        return False
    try:
        pid=int(main_pid)
    except (TypeError,ValueError):
        return False
    if pid<=1:
        return False
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    return True


def stop_owned_controller(output, main_pid, service_result=None):
    output=Path(output).resolve()
    if (output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_')
            or not (output/'.controller.lock').is_file()):
        raise ValueError('Stop only an owned cumulative direct-PD controller')
    # With OOMPolicy=stop, systemd can stop the still-live supervisor after
    # killing its native child. MAINPID alone cannot identify a manual stop.
    # Preserve failure evidence and any existing explicit marker; never turn
    # a failed service result into a new user pause.
    if service_result and service_result != 'success':
        return dict(action='service_failure_preserved',service_result=service_result,
                    stop_marker_created=False)
    # A successful plateau hand-off or an unexpected process exit must retain
    # its diagnosis/recovery state. Neither is an instruction from the user to
    # pause. The systemd service manual explicitly says MAINPID is unset after
    # the main process exits, even though ExecStop is still invoked.
    if not main_process_alive(main_pid):
        return dict(action='main_process_already_exited',stop_marker_created=False)
    state=json.loads((output/'status.json').read_text())
    if state.get('state') == 'completed_model_saved':
        return dict(action='completed_model_preserved',stop_marker_created=False)
    created=not (output/'STOP').exists()
    if state.get('state') != 'completed_model_saved' and not (output/'STOP').exists():
        result=subprocess.run(['systemctl','is-system-running'],capture_output=True,text=True,timeout=10)
        write_json(output/'STOP',dict(reason=stop_reason(result.stdout),checkpoint_before_exit=True))
    request_checkpoint_stop(output)
    return dict(action='cooperative_stop_requested',stop_marker_created=created)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(stop_owned_controller(args.output,os.environ.get('MAINPID'),
                                          os.environ.get('SERVICE_RESULT'))))
