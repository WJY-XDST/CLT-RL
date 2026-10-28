"""Run the unchanged static milestone gates for the orientation40 PPO branch.

Only the reviewed source/runtime selector is adapted. Original static10%
baseline, supplemental actual-leg commands, heldout30s, sustained60s and
from-reset full-height steps retain their original validators and criteria.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import sha
import run_static_milestone_validation10 as milestone
import run_static_reset_pair_diagnosis as boundary
import source1201_orientation40 as runtime


def reviewed_source(training_output):
    state = json.loads((Path(training_output)/'status.json').read_text())
    if state['training_active'] or state['state'] != 'agent_review_required':
        raise ValueError('Require the completed200/full27/full17 checkpoint before static validation')
    # Reuse every completed200/full27/full17/source-pin guard in the original
    # selector, substituting only its declared plain stationary profile. The
    # module dependency is restored before constructing any replay command.
    original = boundary.smooth
    boundary.smooth = runtime
    try:
        source, entry = boundary.reviewed_source(training_output)
    finally:
        boundary.smooth = original
    if entry != runtime.SCRIPT:
        raise ValueError('This adapter only validates the declared orientation40 native descendant')
    runtime.require_source(source)
    return source, entry


def run(training_output, output, height_step_pair=None):
    output = Path(output).resolve()
    if (output.parent != boundary.RESULTS or output.exists()
            or not output.name.startswith('joint_pd_static_milestone_')):
        raise ValueError('Use a fresh owned static validation output and preserve earlier evidence')
    original = milestone.reviewed_source
    milestone.reviewed_source = reviewed_source
    try:
        return milestone.run(training_output,output,height_step_pair)
    finally:
        milestone.reviewed_source = original
        if (output.parent == boundary.RESULTS and output.name.startswith('joint_pd_static_milestone_')
                and (output/'plan.json').exists()):
            write_json(output/'orientation40_validator_adapter.json',dict(
                observed_at=datetime.now().astimezone().isoformat(),
                adapter_sha256=sha(__file__),core_milestone_sha256=sha(milestone.__file__),
                original_boundary_selector_sha256=sha(boundary.__file__),
                native_runtime_sha256=sha(runtime.SCRIPT),
                only_changed_dependency='completed-source profile/runtime selector',
                static_thresholds_or_measurement_windows_changed=False,
                model_or_optimizer_updated=False,hardware_enabled=False,policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--height-step-pair',type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.training_output,args.output,args.height_step_pair),indent=2))
