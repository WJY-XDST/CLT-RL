"""Compare saved Gaussian and deterministic replay after a reviewed200 block.

Reuse both complete same51/heldout144 command manifests and deterministic
records. Only the original evaluator's --sample-policy-actions flag changes.
This is a frozen diagnostic, never a training or acceptance entry point.
"""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
LAUNCH_SHA = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sampled_reader():
    """Flip one sampled-flag guard; preserve all numerical reader statements."""
    import review_historical_source3001_conditional_yaw200 as original

    tree = ast.parse(inspect.getsource(original.assess))
    before = ast.dump(tree)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.UnaryOp)
             and isinstance(n.op, ast.Not)
             and ast.unparse(n.operand) == "manifest['sampled_policy_actions']"]
    assert len(nodes) == 1
    changed = copy.deepcopy(tree)
    target = [n for n in ast.walk(changed) if isinstance(n, ast.UnaryOp)
              and ast.unparse(n) == ast.unparse(nodes[0])][0]

    class Replace(ast.NodeTransformer):
        def visit_UnaryOp(self, node):
            return node.operand if node is target else self.generic_visit(node)

    changed = Replace().visit(changed)
    restored = copy.deepcopy(changed)
    matches = [n for n in ast.walk(restored) if isinstance(n, ast.Assert)]
    assert matches
    # Invert this exact leaf back in the original first assertion.
    leaf = [n for n in ast.walk(matches[0]) if isinstance(n, ast.Subscript)
            and ast.unparse(n) == "manifest['sampled_policy_actions']"]
    assert len(leaf) == 1
    replacement_target = leaf[0]

    class Restore(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if node is replacement_target:
                return ast.UnaryOp(op=ast.Not(), operand=node)
            return self.generic_visit(node)

    restored = Restore().visit(restored)
    assert ast.dump(restored) == before
    scope = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(changed), str(original.SCRIPT), 'exec'), scope)
    proof = dict(changed_metadata_guard_nodes=1, original_AST_restored_identical=True,
                 original_metric_threshold_window_survival_statements_unchanged=True)
    return scope['assess'], original.SCRIPT, proof


def assess_sampled(directory):
    import auto_train_mine

    function, reader, proof = sampled_reader()
    captured, prior = [], auto_train_mine.write_json
    try:
        auto_train_mine.write_json = lambda path, value: captured.append((path, value))
        result = function(directory)
    finally:
        auto_train_mine.write_json = prior
    assert len(captured) == 1 and captured[0][1] is result
    result.update(saved_sampling_diagnostic_only=True,
                  reader_metadata_qualification=proof,
                  diagnostic_cannot_accept_a_policy=True)
    write(directory / 'sampled_original7_literal10_descriptive.json', result)
    return result, reader, proof


def components(directory, manifest, cfg):
    import numpy as np

    names = ['time_s', 'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s',
             'left_wheel_rad_s', 'right_wheel_rad_s', 'yaw_rate_rad_s']
    with (directory / 'trace.csv').open() as stream:
        header = stream.readline().strip().split(',')
    count = len(manifest['cases']) * len(manifest['seeds'])
    trace = np.loadtxt(directory / 'trace.csv', delimiter=',', skiprows=1,
                       usecols=[header.index(k) for k in names]).reshape(1000, count, len(names))
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as z:
        means = z['means'][-151:].astype(np.float64)
        initial = z['observations'][0].copy()
    tail = trace[-151:]
    assert np.isfinite(tail).all() and means.shape == (151, count, 6)
    assert abs(tail[0, 0, 0] - 17.) < 1.e-8 and abs(tail[-1, 0, 0] - 20.) < 1.e-8
    factor = cfg['wheel_radius'] / cfg['wheel_track_width']
    limit, scale = cfg['wheel_action_limit'], cfg['action_scale_vel']
    proxy = factor * scale * (means[:, :, 5].clip(-limit, limit)
                             - means[:, :, 2].clip(-limit, limit))
    sampled = factor * (tail[:, :, 2] - tail[:, :, 1])
    wheel = factor * (tail[:, :, 4] - tail[:, :, 3])
    target = np.repeat([c['yaw'] for c in manifest['cases']], len(manifest['seeds']))
    pieces = dict(actor_mean_reference_minus_command=proxy-target,
                  sampled_reference_minus_actor_mean_reference=sampled-proxy,
                  measured_wheel_proxy_minus_sampled_reference=wheel-sampled,
                  body_minus_measured_wheel_proxy=tail[:, :, 5]-wheel)
    error = float(abs(sum(pieces.values()) - (tail[:, :, 5]-target)).max())
    assert error < 1.e-12
    return {k: v.mean(0).tolist() for k, v in pieces.items()}, initial, error


def compare(directory, baseline):
    import numpy as np
    from joint_pd_continuation import sha

    new, reader, proof = assess_sampled(directory)
    old = read(baseline / 'original_and_literal10_conditional_assessment.json')
    am, bm = [read(p / 'manifest.json') for p in (baseline, directory)]
    ac, bc = [read(p / 'resolved_env.json') for p in (baseline, directory)]
    assert not am['sampled_policy_actions'] and bm['sampled_policy_actions']
    assert am['checkpoint_sha256'] == bm['checkpoint_sha256']
    assert am['cases'] == bm['cases'] and am['seeds'] == bm['seeds']
    assert bm['policy_noise_seed'] == 43 and not bm['weights_updated']
    assert not bm['policy_input_remapping'] and not bm['action_bias_probes'] and not bm['wheel_probe']
    assert am['seconds_per_case'] == bm['seconds_per_case'] == 20.
    aa, bb = dict(ac), dict(bc)
    aa.pop('log_dir', None)
    bb.pop('log_dir', None)
    assert aa == bb, 'Never compare different physical/controller configurations'
    x, initial_a, closure_a = components(baseline, am, ac)
    y, initial_b, closure_b = components(directory, bm, bc)
    assert np.array_equal(initial_a, initial_b)
    old_index = {(r['case']['name'], r['seed']): r for r in old['trials']}
    rows = []
    for index, r in enumerate(new['trials']):
        prior = old_index[r['case']['name'], r['seed']]
        assert r['case'] == prior['case']
        rows.append(dict(case=r['case'], seed=r['seed'],
            deterministic_survived=prior['survived'], sampled_survived=r['survived'],
            deterministic_original_passed=prior['original_passed'],
            sampled_original_passed_descriptive_only=r['original_passed'],
            deterministic_strict_passed=prior['original_AND_literal10_passed'],
            sampled_strict_passed_descriptive_only=r['original_AND_literal10_passed'],
            deterministic_metrics=prior['metrics'], sampled_metrics=r['metrics'],
            signed_body_yaw_bias_rad_s=dict(deterministic=prior['yaw_signed_bias_rad_s'],
                                           saved_Gaussian=r['yaw_signed_bias_rad_s']),
            body_yaw_temporal_std_rad_s=dict(deterministic=prior['yaw_standard_deviation_rad_s'],
                                            saved_Gaussian=r['yaw_standard_deviation_rad_s']),
            signed_yaw_error_components_rad_s=dict(
                deterministic={k: v[index] for k, v in x.items()} if prior['survived'] else None,
                saved_Gaussian={k: v[index] for k, v in y.items()} if r['survived'] else None)))
    assert len(rows) == len(old['trials']) == len(new['trials'])
    pins = {str(reader): sha(reader)}
    for folder in (baseline, directory):
        for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'):
            path = folder / name
            pins[str(path)] = sha(path)
    return dict(trials=rows, deterministic_summary=old['summary'],
                sampled_summary_descriptive_only=new['summary'],
                initial_observations_bitwise_equal=True,
                signed_component_closure_max_error=max(closure_a, closure_b),
                original_numeric_reader_metadata_guard_qualification=proof), pins


def run(args):
    grid_marker = args.grid / 'complete_paired_current_stage525.json'
    while not grid_marker.exists():
        raw = subprocess.run(['systemctl', '--user', 'show', args.after_unit,
            '--property=MainPID,SubState,Result'], capture_output=True, text=True, check=True)
        status = dict(row.split('=', 1) for row in raw.stdout.splitlines() if '=' in row)
        assert status['MainPID'] != '0' and status['Result'] == 'success', (
            'Inspect the owned incomplete grid without restarting PPO', status)
        time.sleep(15.)
    import joint_pd_uniform_control_runtime_v3 as runtime
    from joint_pd_continuation import sha, tensor_finite

    assert sha(SCRIPT) == LAUNCH_SHA
    assert not args.output.exists()
    runtime.configure(args.recipe)
    grid, final, review, archive = [read(p) for p in (grid_marker,
        args.training / 'final_checkpoint_verification_readonly.json',
        args.training / 'independent_review/complete_comparison_agent_decision_required.json',
        args.archive_manifest)]
    assert grid['complete'] and final['verified'] and review['complete'] and archive['verified']
    source = Path(final['checkpoint'])
    saved, _ = runtime.require_source(source)
    assert tensor_finite(saved) and sha(source) == final['checkpoint_sha256'] == archive['checkpoint_sha256']
    assert sha(source) == grid['candidate_checkpoint_sha256']
    assert runtime.saved_updates(source) == runtime.recipe()['source_actual_updates'] + 200
    assert runtime.saved_updates(source) == review['candidate_actual_updates'] == archive['actual_updates']
    assert archive['actual_CPU_runner_load_optimizer_true_full_state_bitwise_verified']
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(v['step']) == 20 * runtime.saved_updates(source)
               for v in saved['optimizer_state_dict']['state'].values())
    sampled_reader()  # Qualify the unchanged numerical reader before native work.
    args.output.mkdir()
    pin_paths = [SCRIPT, args.recipe, grid_marker,
        args.training / 'final_checkpoint_verification_readonly.json', source,
        args.training / 'independent_review/complete_comparison_agent_decision_required.json',
        args.archive_manifest, runtime.SCRIPT]
    pins = {str(p): sha(p) for p in pin_paths}
    results = []
    for scope, expected in [('same51', 51), ('heldout144', 144)]:
        baseline = args.training / 'independent_review' / scope
        manifest = read(baseline / 'manifest.json')
        assert manifest['checkpoint_sha256'] == sha(source)
        assert len(manifest['cases']) * len(manifest['seeds']) == expected
        destination = args.output / (scope + '_saved_Gaussian')
        commands = args.output / (scope + '_commands.json')
        write(commands, manifest['cases'])
        argv = runtime.replay_command(source, destination)
        argv[argv.index('--seconds') + 1] = '20.'
        at = argv.index('--seeds') + 1
        argv[at:at+3] = list(map(str, manifest['seeds']))
        assert '--sample-policy-actions' not in argv
        argv += ['--command-probe-json', str(commands), '--sample-policy-actions']
        write(args.output / (scope + '_launch.json'), dict(command=argv,
            checkpoint_sha256=sha(source), saved_std=saved['model_state_dict']['std'].tolist(),
            weights_updated=False, optimizer_executed=False, policy_accepted=False))
        with (args.output / (scope + '.log')).open('x') as log:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result, inputs = compare(destination, baseline)
        results.append(dict(scope=scope, **result))
        pins.update(inputs)
    assert sum(len(r['trials']) for r in results) == 195
    assert all(sha(path) == digest for path, digest in pins.items())
    write(args.output / 'complete_saved_Gaussian_deployment_gap195.json', dict(
        complete=True, observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=sha(source), actual_updates=runtime.saved_updates(source),
        saved_std=saved['model_state_dict']['std'].tolist(), paired_trials=195, scopes=results,
        source_pins=pins, original_numeric_readers_unchanged=True,
        all_baseline_trials_and_failures_retained=True, optimizer_executed=False,
        saved_weights_std_or_optimizer_modified=False, new_PPO_updates=0,
        condition_specific_training_logic=False, acceptance_changed=False,
        next_source_or_parameter_selected=False, policy_accepted=False,
        limitations=['Saved Gaussian is a diagnosis of distribution/mean bias, not deployment acceptance.',
            'Only3 action-noise streams from the original fixed seed43 are repeated across cases;195 trials are not195 independent noise streams.',
            'Signed proxy components close arithmetically, without establishing independent friction/servo/dither causality.',
            'Full speed/yaw coverage, sustained verification and new-policy MuJoCo remain required.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--grid', type=Path)
    parser.add_argument('--archive-manifest', type=Path)
    parser.add_argument('--after-unit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--qualify-reader-only', action='store_true')
    args = parser.parse_args()
    if args.qualify_reader_only:
        _, _, proof = sampled_reader()
        print(json.dumps(proof))
    else:
        assert all((args.recipe, args.training, args.grid, args.archive_manifest,
                    args.after_unit, args.output))
        run(args)
