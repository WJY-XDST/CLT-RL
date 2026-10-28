"""CPU-only feasibility of independent future leg/wheel exploration floors.

This does not select the next trial or checkpoint. Reuse the existing two
projection hooks, retaining exact source/candidate models,17 Adam states and
actual LR on load. No historical/pinned training file is edited; no optimizer
step, simulator, capacity admission or PPO is executed.
"""
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
from types import FunctionType,SimpleNamespace

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

from joint_pd_continuation import ROOT,sha
import joint_pd_AF7001_wheelD075_runtime as runtime
import joint_pd_source200_leg05_runtime as projection
import wheel_legged_gym_isaaclab.exploration as exploration


SCRIPT=Path(__file__).resolve()
TRAINING=ROOT/'mine_model/results/joint_pd_optimization_20261023_192237_032943_fbd005e4_AF7001_wheelD075_200'
OUTPUT=TRAINING/'wheel005_projection_mechanism_only_20261023'
FLOORS=dict(global_minimum_std=.005,minimum_leg_std=.01,minimum_wheel_std=.005)


def qualify():
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    plan_path=TRAINING/'experiment_plan.json'
    final_path=TRAINING/'final_checkpoint_verification_readonly_20261019.json'
    plan,final=[json.loads(path.read_text()) for path in (plan_path,final_path)]
    assert final['verified'] and final['actual_ppo_updates']==7201 and final['all17_Adam_step']==144020
    paths=[SCRIPT,runtime.SCRIPT,Path(projection.__file__),Path(exploration.__file__),plan_path,final_path]
    fixtures=[]
    obs=TensorDict({'policy':torch.linspace(-.2,.2,32*27).reshape(32,27)},batch_size=[32])
    rng=torch.random.get_rng_state().clone()
    # Initialization consumes only private CPU RNG; restore the caller state.
    with torch.random.fork_rng(devices=[]):
        for label,path,actual in [('source7001',Path(plan['source_checkpoint']),7001),
                                  ('candidate7201',Path(final['checkpoint']),7201)]:
            saved,cfg=runtime.require_source(path)
            assert len(saved['optimizer_state_dict']['state'])==17
            assert all(float(v['step'])==actual*20 for v in saved['optimizer_state_dict']['state'].values())
            agent_path=path.parent/'params/agent.yaml';agent=yaml.safe_load(agent_path.read_text())
            options=dict(agent['policy']);assert options.pop('class_name')=='ActorCritic'
            assert agent['algorithm']['class_name']=='PPO'
            assert not agent['algorithm'].get('joint_height_feedback_coef',0.)
            assert not agent['algorithm'].get('joint_mean_guard_coef',0.)
            initial=OUTPUT/label
            initial.mkdir()
            previous_runner,cls=runtime.private_runtime(path).install_runner(initial)
            handles=None;old_enforce=None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    policy=ActorCritic(obs,agent['obs_groups'],6,**options)
                runner=object.__new__(cls)
                runner.alg=SimpleNamespace(policy=policy,optimizer=torch.optim.Adam(policy.parameters()),rnd=None)
                runner.current_learning_iteration=0
                runner.logger_type,runner.disable_logs='tensorboard',True
                runner.load(str(path),load_optimizer=True,map_location='cpu')
                runtime.base.equal_state(saved['model_state_dict'],policy.state_dict(),'loaded_model')
                runtime.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'loaded_all17_Adam')
                prediction=policy.act_inference(obs).detach().clone()
                scope=dict(projection.__dict__,LEG_STD_FLOOR=.01,GLOBAL_STD_FLOOR=.005,WHEEL_STD_FLOOR=.005,
                    profile=lambda:dict(FLOORS,mechanism_only=True,next_trial_selected=False),
                    print=lambda *_args,**_kwargs:None)
                install=FunctionType(projection.install_leg_exploration.__code__,scope,
                    projection.install_leg_exploration.__name__)
                old_enforce=install(initial/'projection_fixture')
                handles=exploration.enforce_positive_exploration(policy,runner.alg.optimizer,.005,minimum_wheel_std=.005)
                identity=policy.std
                runtime.base.equal_state(saved['model_state_dict'],policy.state_dict(),'after_projection_install_model')
                runtime.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'after_projection_install_all17')
                test_rows=[]
                for values in [[-.02,.002,-.001,.004,.007,.002],[.02,.03,.02,.04,.05,.03]]:
                    fixture=torch.tensor(values)
                    with torch.no_grad():policy.std.copy_(fixture)
                    for hook in runner.alg.optimizer._optimizer_step_post_hooks.values():
                        hook(runner.alg.optimizer,(),{})
                    expected=fixture.clamp_min(.005)
                    expected[[0,1,3,4]]=expected[[0,1,3,4]].clamp_min(.01)
                    assert torch.equal(policy.std,expected) and policy.std is identity and identity.requires_grad
                    runtime.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'manual_hooks_preserve_all17')
                    test_rows.append(dict(before=values,expected=expected.tolist(),actual=policy.std.detach().tolist()))
                policy.load_state_dict(saved['model_state_dict'],strict=True)
                assert torch.equal(prediction,policy.act_inference(obs).detach())
                runtime.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'restored_all17_Adam')
                assert all(parameter.requires_grad for parameter in policy.parameters())
                fixtures.append(dict(label=label,checkpoint=str(path),checkpoint_sha256=sha(path),
                    actual_updates=actual,all17_Adam_step=actual*20,saved_std=saved['model_state_dict']['std'].tolist(),
                    actual_saved_LR=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']],
                    initial_model_all17_Adam_and_LR_bitwise_preserved=True,
                    all17_parameters_remain_trainable=True,all_six_std_parameter_identity_preserved=True,
                    inference_before_after_restoring_fixture_bitwise_equal=True,manual_projection_fixtures=test_rows))
            finally:
                if handles is not None:handles.remove()
                if old_enforce is not None:exploration.enforce_positive_exploration=old_enforce
                runners.OnPolicyRunner=previous_runner
            paths.extend((path,path.parent/'params/env.yaml',agent_path))
    assert torch.equal(rng,torch.random.get_rng_state())
    pins={str(path):sha(path) for path in paths}
    value=dict(verified=True,observed_at=datetime.now().astimezone().isoformat(),
        mechanism_only=True,floors_examined=FLOORS,fixtures=fixtures,
        original_projection_function_numeric_code_reused=True,
        no_historical_training_source_file_changed=True,source_pins=pins,
        actual_optimizer_steps=0,actual_new_PPO_updates=0,simulator_started=False,
        next_source_selected=False,next_parameter_change_selected=False,
        production_or_capacity_admission_granted=False,policy_accepted=False,
        conclusion='Existing isolated two-hook mechanism can retain leg floor.01 while allowing wheel floor.005; saved initial six std and all17 Adam remain unchanged. Performance and final source choice remain unproven.')
    assert all(sha(path)==digest for path,digest in pins.items())
    with (OUTPUT/'qualification.json').open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(verified=True,mechanism_only=True,fixtures=len(fixtures),optimizer_steps=0,
        next_trial_selected=False,report=str(OUTPUT/'qualification.json'))))
    return value


if __name__=='__main__':qualify()
