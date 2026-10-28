"""Frozen saved-policy replay with explicit, command-independent stressors.

No optimizer or training entry point is exposed. Nominal controller validation
runs before property perturbations, and the actual PhysX readback is recorded.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import joint_pd_saved_bundle_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.robustness import RobustnessSession, RobustnessSpec


SCRIPT = Path(__file__).resolve()


def replay(recipe, specification, arguments):
    runtime.configure(recipe)
    spec = RobustnessSpec.from_dict(json.loads(specification.read_text()))
    original = runtime.private_for_count

    def private(*args, **kwargs):
        module = original(*args, **kwargs)
        nominal_observer = module.observe_environment

        def observe(env, output):
            nominal_observer(env, output)
            session = RobustnessSession(env.unwrapped, spec)
            env.unwrapped._robustness_session = session
            write_json(Path(output) / 'robustness_contract.json', dict(
                nominal_saved_controller_verified_before_perturbation=True,
                specification=asdict(spec), applied=session.evidence,
                policy_weights_changed=False, optimizer_executed=False,
                stress_test_not_nominal_acceptance=True, policy_accepted=False,
                source_pins={str(p): sha(p) for p in (SCRIPT, specification,
                    Path(sys.modules[RobustnessSession.__module__].__file__))}))

        module.observe_environment = observe
        return module

    runtime.private_for_count = private
    try:
        runtime.replay(arguments)
    finally:
        runtime.private_for_count = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay',))
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--robustness-json', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    replay(args.recipe.resolve(), args.robustness_json.resolve(), remaining)
