"""Build an explicitly declared two-kernel application reward revision.

Only the denominators of the existing positive speed/yaw kernels change.
The task's action maps, physics, negative costs, height/posture terms, clipping,
termination and observation callbacks are retained verbatim. Nothing installs
itself merely by importing this module.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap

from command_relative_tracking_kernel import command_relative_variance, validate_parameters


SCRIPT = Path(__file__).resolve()
TAG = '_declared_command_relative_tracking_kernel'
PROOF = '_declared_command_relative_tracking_proof'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(command_caps, floor_fraction):
    assert set(command_caps) == {'speed', 'yaw'}
    caps = {key:float(value) for key,value in command_caps.items()}
    fraction = float(floor_fraction)
    for cap in caps.values():
        validate_parameters(1.,cap,fraction)
    helper = Path(inspect.getfile(command_relative_variance)).resolve()
    return dict(version='positive_tracking_command_relative_variance_v1',
        formula='sigma_eff = sigma * (command^2 + (cap*f)^2) / (cap^2 + (cap*f)^2)',
        command_caps=caps, common_floor_fraction=fraction,
        same_formula_for_every_environment_height_sign_and_task=True,
        nominal_sigma_retained_at_command_cap=True,
        zero_command_has_finite_absolute_width=True,
        acceptance_threshold_used_by_reward=False,
        helper_sha256=sha(helper), builder_sha256=sha(SCRIPT))


def build_reward(original, command_caps, floor_fraction):
    spec = contract(command_caps,floor_fraction)
    assert original.__name__ == '_get_rewards'
    assert not original.__closure__, 'Preserve a self-contained application method'
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    assert len(tree.body) == 1 and isinstance(tree.body[0],ast.FunctionDef)
    assert not tree.body[0].decorator_list
    initial = ast.dump(tree)
    expected = dict(r_track_lin=('lin_vel_error','tracking_sigma',0,'speed'),
                    r_track_ang=('ang_vel_error','tracking_sigma_ang',1,'yaw'))
    changed = []
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:
            continue
        target=node.targets[0]
        if not isinstance(target,ast.Name) or target.id not in expected:
            continue
        error,sigma,column,axis=expected[target.id]
        wanted=ast.parse(f'torch.exp(-{error} / self.cfg.rewards.{sigma})',mode='eval').body
        assert ast.dump(node.value)==ast.dump(wanted)
        denominator=node.value.args[0]
        assert isinstance(denominator,ast.BinOp) and isinstance(denominator.op,ast.Div)
        before=denominator.right
        replacement=ast.parse('_command_relative_variance(self._commands[:, '
            +str(column)+'], self.cfg.rewards.'+sigma+', '
            +repr(spec['command_caps'][axis])+', '+repr(spec['common_floor_fraction'])+')',
            mode='eval').body
        denominator.right=replacement
        changed.append((denominator,before,replacement,target.id))
    assert len(changed)==2 and {name for *_,name in changed}==set(expected)
    modified=copy.deepcopy(tree)
    for owner,before,_,_ in changed:
        owner.right=before
    assert ast.dump(tree)==initial, 'A change escaped the two declared denominators'
    namespace=dict(original.__globals__,_command_relative_variance=command_relative_variance)
    exec(compile(ast.fix_missing_locations(modified),str(SCRIPT),'exec'),namespace)
    revised=namespace['_get_rewards']
    revised.__qualname__=original.__qualname__
    setattr(revised,TAG,spec)
    proof=dict(verified=True,contract=spec,modified_positive_kernel_denominators=sorted(expected),
        all_other_application_reward_AST_identical=True,
        original_AST_sha256=hashlib.sha256(initial.encode()).hexdigest(),
        revised_AST_sha256=hashlib.sha256(ast.dump(modified).encode()).hexdigest(),
        original_application_source=inspect.getfile(original),
        original_application_source_sha256=sha(inspect.getfile(original)),
        new_reward_terms_added=0,observation_callbacks_added=0,
        action_or_physics_or_sampler_changed=False,ordinary_PPO_loss_changed=False,
        acceptance_changed=False,policy_accepted=False)
    setattr(revised,PROOF,proof)
    return revised,proof,modified


def install_class_reward(env_class, command_caps, floor_fraction):
    """Install once on the task class in an explicitly scoped simulation process."""
    spec=contract(command_caps,floor_fraction)
    current=env_class._get_rewards
    previous=getattr(current,TAG,None)
    if previous is not None:
        assert previous==spec, 'Do not mix reward profiles in one simulation'
        return getattr(current,PROOF)
    revised,proof,_=build_reward(current,command_caps,floor_fraction)
    env_class._get_rewards=revised
    assert getattr(env_class._get_rewards,TAG)==spec
    return proof
