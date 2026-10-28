"""Observe the existing bounded guide fixture at every physics substep.

No policy is loaded or trained. The original guide, targets, physics and
feasibility checks are unchanged. Native motor torque is not inferred from
the sampled explicit effort or from the algebraic PD demand.
"""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'wheel_legged_isaaclab/scripts/check_direct_joint_pd.py'
OUTPUT = ROOT / 'mine_model/results/joint_pd_control_chain_audit_20261023/native_substep_observer_v2'
SAMPLES = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install(cls, args):
    original = cls._apply_action

    def observed(self):
        original(self)
        import torch
        q = self._robot.data.joint_pos[:, self._leg_joint_ids]
        dq = self._robot.data.joint_vel[:, self._leg_joint_ids]
        reference = self._joint_position_ref_applied
        ff = self._leg_gravity_effort_applied
        p = self.cfg.leg_joint_stiffness * (reference - q)
        d = -self.cfg.leg_joint_damping * dq
        SAMPLES.append(dict(physics_counter=self._sim_step_counter, policy_step=(self._sim_step_counter - 1) // self.cfg.decimation, substep=(self._sim_step_counter - 1) % self.cfg.decimation, q_rad=q.cpu().tolist(), dq_rad_s=dq.cpu().tolist(), reference_rad=reference.cpu().tolist(), gravity_feedforward_Nm=ff.cpu().tolist(), algebraic_P_demand_Nm=p.cpu().tolist(), algebraic_D_demand_Nm=d.cpu().tolist(), algebraic_unclipped_total_demand_Nm=(p + d + ff).cpu().tolist(), finite=bool(torch.isfinite(q).all() and torch.isfinite(dq).all() and torch.isfinite(ff).all())))
    cls._apply_action = observed

def finalize(scope, error=None):
    import numpy as np
    args = scope.get('args')
    if args is not None and SAMPLES:
        reference = np.asarray([x['reference_rad'] for x in SAMPLES])
        effort = np.asarray([x['gravity_feedforward_Nm'] for x in SAMPLES])
        groups = {}
        for (index, row) in enumerate(SAMPLES):
            groups.setdefault(row['policy_step'], []).append(index)
        four = [indices for indices in groups.values() if len(indices) == 4]
        unchanged = max((float(np.max(np.abs(reference[indices] - reference[indices[0]]))) for indices in four), default=0.0)
        changing = sum((float(np.max(np.abs(effort[indices] - effort[indices[0]]))) > 1e-07 for indices in four))
        expected = round(args.seconds / 0.005)
        valid = error is None and len(SAMPLES) == expected and (len(four) == round(args.seconds / 0.02)) and (unchanged == 0.0) and (changing > 0) and all((row['finite'] for row in SAMPLES))
        result = dict(verified=valid, observed_at=datetime.now().astimezone().isoformat(), mode=args.mode, seconds=args.seconds, physics_substep_calls=len(SAMPLES), expected_substep_calls=expected, full_four_substep_policy_groups=len(four), constant_target_max_difference_rad=unchanged, groups_with_feedforward_changes_inside_held_target=changing, gravity_feedforward_peak_abs_Nm=float(np.abs(effort).max()), error=error, original_fixture_and_numeric_checks_AST_restored_identical=True, observer_assigns_actions_states_or_forces=False, weights_loaded=False, optimizer_executed=False, production_training_modified=False, policy_accepted=False, native_motor_torque_measured=False, source_pins={str(p): sha(p) for p in [Path(__file__), ORIGINAL, ROOT / 'IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py', ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py']}, limitations=['Single world-slider guide fixture, not free balancing or policy acceptance.', 'Algebraic PD demand is an estimate; no native implicit motor torque is claimed.', 'No full mechanical action-envelope or continuous-time collision safety claim.'], finalized_before_SimulationApp_close=True)
        (OUTPUT / 'native_substep_observer_samples.json').write_text(json.dumps(SAMPLES) + '\n')
        (OUTPUT / 'native_substep_observer.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for (k, v) in result.items() if k not in ('source_pins', 'limitations')}), flush=True)
        if error is None and (not valid):
            raise RuntimeError('Inspect the saved substep observations; the observer checks failed')

def main():
    OUTPUT.mkdir(exist_ok=False)
    tree = ast.parse(ORIGINAL.read_text())
    before = ast.dump(tree)
    entry = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
    matches = [i for (i, n) in enumerate(entry.body) if isinstance(n, ast.ClassDef) and n.name == 'ProbeEnv']
    assert len(matches) == 1
    index = matches[0] + 1
    added = ast.parse('_install_substep_audit(ProbeEnv, args)').body[0]
    entry.body.insert(index, added)
    shutdown = tree.body[-1]
    assert isinstance(shutdown, ast.Try) and len(shutdown.finalbody) == 1
    assert ast.unparse(shutdown.finalbody[0]) == 'app.close()'
    finalizer = ast.parse("_finalize_substep_audit(globals(), None if __import__('sys').exc_info()[0] is None else repr(__import__('sys').exc_info()[1]))").body[0]
    shutdown.finalbody.insert(0, finalizer)
    code = compile(ast.fix_missing_locations(tree), str(ORIGINAL), 'exec')
    shutdown.finalbody.pop(0)
    entry.body.pop(index)
    assert ast.dump(tree) == before
    scope = dict(__name__='__main__', __file__=str(ORIGINAL), _install_substep_audit=install, _finalize_substep_audit=finalize)
    exec(code, scope)
if __name__ == '__main__':
    main()
