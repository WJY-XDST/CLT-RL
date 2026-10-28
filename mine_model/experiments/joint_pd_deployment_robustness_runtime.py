"""Frozen replay of a passed candidate under a separately qualified stress plan.

Only the selected candidate's nominal runtime constructs the environment and
checks its controller. Perturbations are installed afterwards. No training,
optimizer, hardware output or policy promotion entry point is exposed.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def authorized_profile(plan_path, profile, recipe, runtime_module, arguments):
    plan = json.loads(plan_path.read_text())
    diagnostic = (plan.get('schema') == 'diagnostic_deployment_robustness_suite_v1'
                  and plan.get('diagnostic_preacceptance') is True
                  and bool(plan.get('diagnostic_authorization')))
    if (not plan.get('qualified') or (not diagnostic and
            plan.get('schema') != 'post_acceptance_deployment_robustness_suite_v1')):
        raise ValueError('A qualified post-acceptance plan is required')
    if not diagnostic and (plan.get('nominal735_passed') is not True or plan.get('independent189_passed') is not True):
        raise ValueError('Both unchanged nominal acceptance scopes must pass first')
    if runtime_module != plan['nominal_runtime_module'] or recipe.resolve() != Path(plan['recipe']).resolve():
        raise ValueError('The nominal runtime and recipe must match the qualified candidate')
    checkpoint = Path(arguments[arguments.index('--checkpoint') + 1]).resolve()
    if checkpoint != Path(plan['checkpoint']).resolve() or sha(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('Stress testing a different checkpoint is not authorized by this plan')
    for path, digest in plan['source_pins'].items():
        if sha(path) != digest:
            raise ValueError('Changed stress-suite input: ' + path)
    entries = [item for item in plan['profiles'] if item['name'] == profile]
    if len(entries) != 1:
        raise ValueError('Exactly one declared stress profile must be selected')
    return plan, entries[0]


def replay(*, plan_path, profile, recipe, runtime_module, arguments):
    plan, entry = authorized_profile(plan_path, profile, recipe, runtime_module, arguments)
    # Imports that can load the simulator/runtime occur only after the plan gate.
    if not runtime_module.startswith('joint_pd_') or not runtime_module.endswith('_runtime'):
        raise ValueError('A local saved-policy runtime is required')
    located = importlib.util.find_spec(runtime_module)
    if not located or Path(located.origin).resolve().parent != ROOT / 'mine_model/experiments':
        raise ValueError('The nominal runtime must be in this project')
    from auto_train_mine import write_json
    from wheel_legged_gym_isaaclab.deployment_randomization import (
        DeploymentRandomizationSpec, DeploymentRandomizationSession,
    )
    specification = Path(entry['specification'])
    spec = DeploymentRandomizationSpec.from_dict(json.loads(specification.read_text()))
    runtime = importlib.import_module(runtime_module)
    runtime.configure(recipe)
    original = runtime.private_for_count
    namespace = runtime.replay.__globals__
    if namespace.get('private_for_count') is not original:
        raise RuntimeError('Inspect the actual nominal replay observer binding before execution')

    def private(*args, **kwargs):
        module = original(*args, **kwargs)
        nominal_observer = module.observe_environment

        def observe(env, output):
            nominal_observer(env, output)
            session = DeploymentRandomizationSession(env.unwrapped, spec)
            env.unwrapped._deployment_randomization_session = session
            write_json(Path(output) / 'robustness_contract.json', dict(
                nominal_saved_controller_verified_before_perturbation=True,
                specification=asdict(spec), applied=session.evidence,
                policy_weights_changed=False, optimizer_executed=False,
                stress_test_not_nominal_acceptance=True, policy_accepted=False,
                qualified_suite=str(plan_path), profile=profile,
                nominal_checkpoint_sha256=plan['checkpoint_sha256'],
                source_pins={str(path): sha(path) for path in
                    (SCRIPT, specification, plan_path, Path(importlib.import_module(
                        DeploymentRandomizationSession.__module__).__file__))}))

        module.observe_environment = observe
        return module

    namespace['private_for_count'] = private
    try:
        runtime.replay(arguments)
    finally:
        namespace['private_for_count'] = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay',))
    parser.add_argument('--suite-plan', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--nominal-runtime-module', required=True)
    args, remaining = parser.parse_known_args()
    # Keep --recipe for the existing evaluator command contract, though the
    # nominal replay function itself receives its configuration via configure.
    replay(plan_path=args.suite_plan.resolve(), profile=args.profile,
           recipe=args.recipe.resolve(), runtime_module=args.nominal_runtime_module,
           arguments=remaining)
