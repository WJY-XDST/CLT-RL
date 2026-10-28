"""Original reward audits plus explicit reconstruction of the one new component."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

from auto_train_mine import write_json
import audit_stationary_height_prefix as original_prefix
import diagnose_reward_balance as original_replay
from joint_pd_continuation import ROOT, sha

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_stationary_twist_runtime as twist


def cost(velocity, yaw, command):
    error = np.concatenate((velocity / .025, ((yaw - command) / .025)[..., None]), axis=-1)
    absolute = abs(error)
    return np.where(absolute < 1., .5 * error**2, absolute - .5).sum(axis=-1)


def audit_replay(directory):
    directory = Path(directory)
    result = original_replay.audit(directory)
    cfg = json.loads((directory / 'resolved_env.json').read_text())
    if cfg['rewards']['standing_velocity'] != -1.:
        raise ValueError('Require the explicit normalized stationary component')
    a = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    manifest = json.loads((directory / 'manifest.json').read_text())
    dt = manifest['policy_dt']
    # Fixed cases are zero or nonzero forward targets. Mask directly from the
    # reached per-step command; ignore early ramps of moving cases because the
    # separate target_lin_vel_x is not recorded in the CSV.
    static_ids = [i for i, c in enumerate(manifest['cases']) if c['speed'] == 0.]
    mask = np.isin(a['case_id'], static_ids)
    velocity = np.stack((a['speed_m_s'], a['lateral_speed_m_s']), axis=-1)
    expected = np.clip(-cost(velocity, a['yaw_rate_rad_s'], a['yaw_cmd_rad_s']), -5., 5.) * dt
    error = float(abs(expected[mask] - a['reward_standing_velocity'][mask]).max())
    if error > 1.e-6:
        raise ValueError('Actual stationary reward differs from its declared profile')
    result.update(stationary_twist_profile=twist.profile(),
        stationary_twist_independent_reconstruction_max_error=error,
        stationary_twist_reconstructed_samples=int(mask.sum()),
        audit_script_sha256=sha(Path(__file__)))
    return result


def audit_prefix(output):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original_prefix.audit)))
    before = copy.deepcopy(tree)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
            matches.extend(k for k in node.keywords if k.arg == 'standing_velocity')
    if len(matches) != 1:
        raise ValueError('Requalify the original complete native reward auditor')
    original_value = copy.deepcopy(matches[0].value)
    matches[0].value = ast.parse('_twist_cost(v, yaw, c[..., 1])', mode='eval').body
    restored = copy.deepcopy(tree)
    restored_match = [k for n in ast.walk(restored) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'dict'
        for k in n.keywords if k.arg == 'standing_velocity']
    restored_match[0].value = original_value
    if ast.dump(restored) != ast.dump(before):
        raise ValueError('Native auditor changed outside the one component formula')
    ast.fix_missing_locations(tree)
    namespace = dict(vars(original_prefix), _twist_cost=cost, __file__=str(Path(__file__).resolve()))
    exec(compile(tree, str(Path(__file__)) + ':private_original_audit', 'exec'), namespace)
    return namespace['audit'](output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--native-prefix', action='store_true')
    args = parser.parse_args()
    result = audit_prefix(args.directory) if args.native_prefix else audit_replay(args.directory)
    if not args.native_prefix:
        write_json(args.directory / 'stationary_twist_reward_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'conditions')}))
