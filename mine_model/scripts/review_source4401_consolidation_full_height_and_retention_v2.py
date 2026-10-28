"""Complete the shared-helper metadata namespace; retain the failed v1 proof intact."""
import argparse
import json
from pathlib import Path

import review_source4401_consolidation_full_height_and_retention as previous
import plain27_sampler_metadata_adapter as adapter
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
STATIC_QUALIFICATION = RESULTS / 'joint_pd_source4401_consolidation_static_qualification_v2_20261019.json'
_module = adapter.private_module(previous.__file__, '_source4401_consolidation_readers_v2')
_module.SCRIPT, _module.STATIC_QUALIFICATION = SCRIPT, STATIC_QUALIFICATION
_module.ADAPTER_EVIDENCE = _module.EVIDENCE
_module._api.SCRIPT, _module._api.STATIC_QUALIFICATION = SCRIPT, STATIC_QUALIFICATION
_previous_retention = _module.retention


def retention():
    module = _previous_retention()
    original = module.dependencies

    def dependencies():
        return dict(original(), **{str(Path(previous.__file__)): sha(previous.__file__)})

    module.dependencies = dependencies
    return module


_module.retention = _module._api.retention = retention


def qualify(launch_path, checkpoint, qualification):
    result = _module.qualify(launch_path, checkpoint, qualification)
    result['source_pins'][str(Path(previous.__file__))] = sha(previous.__file__)
    result.update(shared_helper_metadata_namespace_completed=True,
        v1_partial_qualification_not_used=True, original_numeric_readers_unchanged=True)
    write_json(Path(qualification), result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--outputs', type=Path, nargs=4)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else _module.run(args.training_launch, args.qualification,
            args.outputs, args.static_output))
    print(json.dumps({k: result[k] for k in ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
