"""A reviewed block must advance automatically without resetting the policy."""
import pytest

from continue_reviewed_joint_pd import entropy_command,next_entropy


def test_entropy_revision_preserves_the_exact_source_and_environment_command():
    command=['python','train.py','--checkpoint_path','latest.pt','--num_envs','8192',
             '--min_exploration_std','.01','env.leg_control_mode="implicit_joint_position"']
    before=command.copy();revised=entropy_command(command,.005)
    assert command==before and revised[:-1]==before
    assert revised[-1]=='agent.algorithm.entropy_coef=0.005'


@pytest.mark.parametrize('value',[0.,-.1,.021,float('nan'),float('inf'),True])
def test_unbounded_exploration_revision_is_rejected(value):
    with pytest.raises(ValueError):entropy_command([],value)


def test_reset_flags_are_never_allowed_in_a_continuation():
    for flag in ('--reset_optimizer','--reset_exploration_std'):
        with pytest.raises(ValueError):entropy_command([flag],.005)


def test_survival_regression_reduces_exploration_without_rolling_back_weights():
    history=[dict(survivors=27,height_error_fraction=.3),dict(survivors=21,height_error_fraction=.3)]
    assert next_entropy(history,.005)==.0025


def test_repeated_height_plateau_adapts_exploration_within_its_bound():
    history=[dict(survivors=21,height_error_fraction=v) for v in (.315,.314,.316)]
    assert next_entropy(history,.005)==pytest.approx(.0075)
    assert next_entropy(history,.02)==.02


def test_improving_height_does_not_trigger_another_hyperparameter_change():
    history=[dict(survivors=21,height_error_fraction=v) for v in (.315,.25,.18)]
    assert next_entropy(history,.005)==.005


def test_review_boundary_automatically_starts_the_next_block(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import yaml
    import continue_reviewed_joint_pd as controller
    root=tmp_path;results=root/'mine_model/results';results.mkdir(parents=True)
    previous=results/'source';previous.mkdir()
    base=results/'coordinator'
    def report():
        names=['stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
               'spin_left','spin_right','height_low','height_high']
        return dict(total_trials=27,survivors=21,resets=6,passed=False,conditions=[
            dict(name=name,height=.225 if name=='height_low' else .375 if name=='height_high' else .3,
                 trials=[dict(survived=True,metrics=dict(height_mae_m=.07)) for _ in range(3)]) for name in names])
    def boundary(directory,actual):
        (directory/'replay').mkdir(parents=True)
        (directory/'params').mkdir()
        (directory/'params/agent.yaml').write_text(yaml.safe_dump(dict(algorithm=dict(class_name='PPO',entropy_coef=.005))))
        (directory/'status.json').write_text(json.dumps(dict(actual_ppo_updates=actual,checkpoint=str(directory/'model.pt'))))
        (directory/'replay/assessment.json').write_text(json.dumps(report()))
    boundary(previous,501)
    calls=[]
    def train_and_review(args,prepare_function):
        calls.append(args)
        args.output.mkdir()
        boundary(args.output,501+len(calls)*100)
        if len(calls)==2:(base/'STOP').touch()
    monkeypatch.setattr(controller,'ROOT',root)
    # Record the original dependency so the run's scoped command adapter is
    # restored after the test as well.
    monkeypatch.setattr(controller.block,'continuation_command',controller.block.continuation_command)
    monkeypatch.setattr(controller.block,'run',train_and_review)
    controller.run(previous,base,'owned.service')
    assert len(calls)==2
    assert calls[1].previous==calls[0].output
    assert all(item.updates==100 for item in calls)
    assert json.loads((base/'coordinator.json').read_text())['state']=='explicit_stop_respected'
