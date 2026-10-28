"""Quantify absolute-kernel sensitivity at the original relative criterion.

The entire original matrix is kept. The counterfactual uses one smooth global
formula for both tracking axes and does not run a policy or change a reward.
Numbers at the threshold assume zero error on other axes and unit quality;
they are kernel sensitivities, not predicted PPO learning or acceptance.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from command_relative_tracking_kernel import command_relative_variance, validate_parameters


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(matrix, training_plan, output, fraction):
    assert not output.exists()
    complete = read(matrix / 'complete_frozen_command_matrix.json')
    original_scope = read(matrix / 'qualification.json')
    assert complete['complete'] and len(complete['all_trials']) == original_scope['trial_count'] == 735
    assert original_scope['unique_numeric_commands'] == 219
    spec = read(training_plan)
    assert spec['ordinary_full_policy_PPO'] and not spec['auxiliary_losses_enabled']
    cfg = spec['training_config']
    acceptance = ROOT / 'mine_model/config/tracking_acceptance.json'
    criteria = read(acceptance)
    # This is the criterion used for this audit only, never a live reward input.
    relative = criteria['relative_error']
    assert relative == .1
    caps = dict(speed=max(abs(v) for key in ('ranges_lin_vel_x', 'ranges_reverse_lin_vel_x')
                          for v in cfg['commands'][key]) * cfg['commands']['training_motion_scale'],
                yaw=max(abs(v) for v in cfg['commands']['ranges_ang_vel_yaw'])
                    * cfg['commands']['training_motion_scale'])
    sigmas = dict(speed=cfg['rewards']['tracking_sigma'], yaw=cfg['rewards']['tracking_sigma_ang'])
    coefficients = dict(speed=cfg['rewards']['tracking_lin_vel'], yaw=cfg['rewards']['tracking_ang_vel'])
    for axis in caps:
        validate_parameters(sigmas[axis],caps[axis],fraction)
    source = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
    tree = ast.parse(source.read_text())
    expressions = {}
    expected = dict(r_track_lin='torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)',
                    r_track_ang='torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_ang)')
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in expected:
                    assert target.id not in expressions
                    expressions[target.id]=ast.unparse(node.value)
                    assert ast.dump(node.value)==ast.dump(ast.parse(expected[target.id],mode='eval').body)
    assert expressions.keys()==expected.keys()
    unique = {}
    failed_magnitudes = {axis:[] for axis in caps}
    for trial in complete['all_trials']:
        command = trial['case']
        key=tuple(command[a] for a in ('speed','yaw','height'))
        unique[key]=command
        for axis in caps:
            if trial['nonzero_relative_percent'][axis] is not None and trial['nonzero_relative_percent'][axis]>=10.:
                failed_magnitudes[axis].append(abs(command[axis]) / caps[axis])
    assert len(unique)==219
    rows=[]
    max_numeric_error=0.
    for command in unique.values():
        values={}
        for axis in caps:
            cmd=command[axis]
            assert abs(cmd)<=caps[axis]+1.e-8
            old=sigmas[axis]
            new=command_relative_variance(cmd,old,caps[axis],fraction)
            assert new>0 and new<=old+1.e-12
            assert new==command_relative_variance(-cmd,old,caps[axis],fraction)
            assert abs(command_relative_variance(caps[axis],old,caps[axis],fraction)-old)<1.e-12
            error=relative*abs(cmd)
            old_loss=1.-math.exp(-error**2/old)
            new_loss=1.-math.exp(-error**2/new)
            old_gradient=2.*error/old*math.exp(-error**2/old)
            new_gradient=2.*error/new*math.exp(-error**2/new)
            epsilon=1.e-7
            numerical=(math.exp(-(error-epsilon)**2/new)-math.exp(-(error+epsilon)**2/new))/(2.*epsilon)
            max_numeric_error=max(max_numeric_error,abs(numerical-new_gradient))
            values[axis]=dict(command=cmd,command_fraction_of_cap=abs(cmd)/caps[axis],
                nominal_variance=old,counterfactual_variance=new,
                absolute_error_at_original_relative_threshold=error,
                old_positive_kernel_loss_fraction_at_threshold=old_loss,
                normalized_positive_kernel_loss_fraction_at_threshold=new_loss,
                old_positive_reward_loss_rate_at_unit_quality=coefficients[axis]*old_loss,
                normalized_positive_reward_loss_rate_at_unit_quality=coefficients[axis]*new_loss,
                old_kernel_abs_error_derivative_at_threshold=old_gradient,
                normalized_kernel_abs_error_derivative_at_threshold=new_gradient,
                zero_command_relative_error_not_defined=(cmd==0.))
        rows.append(dict(command=command,axes=values))
    assert max_numeric_error<1.e-7
    summary={}
    for axis in caps:
        nonzero=[r['axes'][axis] for r in rows if r['axes'][axis]['command']]
        summary[axis]=dict(unique_nonzero_commands=len(nonzero),
            all_original_failed_trial_count=len(failed_magnitudes[axis]),
            largest_failed_command_fraction_of_cap=max(failed_magnitudes[axis],default=None),
            nominal_variance=sigmas[axis],cap=caps[axis],
            zero_command_counterfactual_absolute_width=math.sqrt(command_relative_variance(0.,sigmas[axis],caps[axis],fraction)),
            smallest_nonzero=min(nonzero,key=lambda v:abs(v['command'])),
            maximum_command=max(nonzero,key=lambda v:abs(v['command'])))
    helper=ROOT/'mine_model/experiments/command_relative_tracking_kernel.py'
    pins={str(f):sha(f) for f in [SCRIPT,helper,source,training_plan,acceptance,
           matrix/'qualification.json',matrix/'complete_frozen_command_matrix.json']}
    report=dict(verified=True,observed_at=datetime.now(timezone.utc).isoformat(),
        original_trials_retained=735,unique_numeric_commands=219,
        active_application_positive_kernel_AST_verified=expressions,
        original_relative_threshold_used_for_audit_only=relative,
        proposed_global_formula='sigma_eff = sigma * (command^2 + (cap*f)^2) / (cap^2 + (cap*f)^2)',
        common_floor_fraction_for_both_axes=fraction,command_caps_from_saved_global_training_configuration=caps,
        same_formula_for_all_heights_speeds_signs_and_groups=True,
        maximum_command_variance_retained=True,zero_command_finite=True,
        analytic_derivative_central_difference_max_error=max_numeric_error,
        summary=summary,all_unique_numeric_commands=rows,source_pins=pins,
        parameter_change_selected=False,policy_executed=False,optimizer_executed=False,
        simulator_started=False,training_modified=False,acceptance_changed=False,policy_accepted=False,
        limitations=['Hypothetical isolated-axis kernel loss at the threshold is not the full recorded reward.',
            'Height/pose quality, negative penalties, Gaussian exploration, critic fit and plant dynamics remain relevant.',
            'This algebraic audit cannot prove convergence or acceptance; any candidate requires paired native training and unchanged full-scope replay.'])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,summary=summary,training_modified=False),ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--training-plan',type=Path,required=True)
    parser.add_argument('--floor-fraction',type=float,default=.25)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.matrix.resolve(),args.training_plan.resolve(),args.output.resolve(),args.floor_fraction)
