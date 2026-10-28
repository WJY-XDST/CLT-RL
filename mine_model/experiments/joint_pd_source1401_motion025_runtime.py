"""Retained plain27 PPO entry for the first review-controlled motion stage.

Only commands.training_motion_scale changes0->.25. Four motor-angle actions,
implicit PD, gravity effort, reset pose/noise, rewards, exploration and all17
Adam states retain the independently validated stationary source's meaning.
"""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys

from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import source1201_orientation40 as previous
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'ac3e1fe3552b5cd27d7ef82f49078f40eedc2db8cbea7a37fdeb980aae39fa1b'
MOTION_SCALE = .25
FLAG = '--begin_verified_motion025'
KEY = previous.KEY
base = previous.base
STATIC = ROOT/'mine_model/results/joint_pd_static_milestone_20261017_source1401/static_milestone.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1401_leg05_smooth10_orientation40_motion025_v1',
        native_runtime_sha256=sha(SCRIPT),previous_runtime_sha256=sha(previous.SCRIPT),
        training_motion_scale=MOTION_SCALE,
        stationary_anchor_sha256=ANCHOR_SHA,
        authority='user_gradual_velocity_only_after_complete_independent_static_milestone')
    return result


def static_ready():
    """Native learning cannot begin from the anchor on a partial static pass."""
    result = json.loads(STATIC.read_text())
    if (not result['static_milestone_verified'] or result['checkpoint_sha256'] != ANCHOR_SHA
            or not result['independent_heldout_and_sustained_passed']
            or not result['repeated_from_reset_height_steps_passed']
            or not result['sampled_exploration_survives_without_reset']
            or result['acceptance_revision'] != acceptance.profile()):
        raise ValueError('Require complete original10% static/30s/60s/full-height-step validation')
    sustained = json.loads(Path(result['sustained_validation']).read_text())
    if (sustained['checkpoint_sha256'] != ANCHOR_SHA or not sustained['motion_release_allowed']
            or sustained['source_actual_updates'] != 1401 or sustained['source_all17_Adam_step'] != 28020):
        raise ValueError('The independent sustained static proof differs from source1401')
    for item in sustained['stages'].values():
        if not item['static_passed'] or any(sha(p) != digest for p,digest in item['original_data_sha256'].items()):
            raise ValueError('A completed static validation or its native evidence changed')
    for label,path in result['height_step_reports'].items():
        report = json.loads(Path(path).read_text())
        sampled = label == 'height_steps_sampled'
        if (report['checkpoint_sha256'] != ANCHOR_SHA or report['survivors'] != 9 or report['resets']
                or report['sampled_policy_actions'] is not sampled
                or report['reset_velocity_amplitude'] != .05
                or (not sampled and not report['height_steps_passed'])
                or report['criteria'] != acceptance.profile()['criteria']
                or any(sha(p) != digest for p,digest in report['source_files'].items())):
            raise ValueError('The full-range deterministic/noisy static step evidence changed')
    return result


def _motion_guard(module,name,expression):
    """Adapt one declared metadata invariant; restore-and-compare the whole AST."""
    function = getattr(module,name)
    tree = ast.parse(inspect.getsource(function))
    original = copy.deepcopy(tree)
    expected = ast.dump(ast.parse(expression,mode='eval').body)
    matches = [n for n in ast.walk(tree) if isinstance(n,ast.Compare)
        and len(n.ops)==1 and isinstance(n.ops[0],ast.NotEq)
        and ast.dump(n.left)==expected and len(n.comparators)==1
        and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value==0.]
    if len(matches) != 1:
        raise ValueError('Requalify the one explicit motion metadata guard: '+name)
    matches[0].comparators[0].value = MOTION_SCALE
    restored = copy.deepcopy(tree)
    check = [n for n in ast.walk(restored) if isinstance(n,ast.Compare)
             and ast.dump(n.left)==expected]
    if len(check) != 1:
        raise ValueError('Ambiguous restored motion invariant')
    check[0].comparators[0].value = 0.
    if ast.dump(restored) != ast.dump(original):
        raise ValueError('Native execution changed outside the declared metadata guard')
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(function),'exec'),module.__dict__)


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1401_motion025_private',base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _motion_guard(module,'require_source',"cfg['commands']['training_motion_scale']")
    _motion_guard(module,'observe_environment','cfg.commands.training_motion_scale')
    original_require,original_observe = module.require_source,module.observe_environment
    transition_sha = None
    if source is not None:
        _,cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1401
                or cfg['commands']['training_motion_scale'] != 0.):
            raise ValueError('Only exact independently validated source1401 starts motion025')
        transition_sha = ANCHOR_SHA
    def require(path,*,initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never repeat the original model200 rollback')
        if transition_sha is not None and sha(path)==transition_sha:
            return previous.require_source(path)
        saved,cfg = original_require(path,initial_transfer=False)
        if not previous.recipe(cfg['rewards']):
            raise ValueError('Preserve the qualified orientation40/smooth10 reward recipe')
        return saved,cfg
    def observe(env,output):
        if not previous.recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Native rewards changed during the sampler-only motion stage')
        result = original_observe(env,output)
        previous.previous.install_failure_capture(env.unwrapped,output)
        return result
    module.profile,module.require_source,module.observe_environment = profile,require,observe
    return module


def require_source(source,transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_source1401_same_leg05_exploration',previous.previous.previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile = profile
    return module.install_leg_exploration(output)


def native(output,arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        static_ready()
    source = Path(arguments[arguments.index('--checkpoint_path')+1])
    require_source(source,transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep the original plain27 ActorCritic')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    install_leg_exploration(output)
    import train_joint_pd_observed as observed
    from joint_pd_review_interval200 import install_observer
    from auto_train_mine import write_json
    write_json(Path(output).parent/'review_interval_runtime.json',install_observer(observed))
    observed.run(output,previous.previous.previous.tires.arguments(arguments))


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint')+1])
    require_source(source,transition=transition)
    if transition:
        if '--overrides_json' not in arguments:
            raise ValueError('Frozen initial transfer must declare its sampler-only metadata override')
        overrides = json.loads(Path(arguments[arguments.index('--overrides_json')+1]).read_text())
        if overrides != {'commands.training_motion_scale':MOTION_SCALE}:
            raise ValueError('Only the motion metadata field changes during frozen transfer')
    elif '--overrides_json' in arguments:
        raise ValueError('Native descendants replay their exact saved contract')
    sys.argv = [str(SCRIPT),*previous.previous.previous.tires.arguments(arguments)]
    tree,original = previous.previous.evaluator_tree()
    exec(compile(tree,str(original),'exec'),dict(__name__='__main__',__file__=str(original),
        _observe_source200=private_runtime().observe_environment))


def command(source,*args,**kwargs):
    argv,final,saved = previous.command(source,*args,**kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    if sha(source)==ANCHOR_SHA:
        argv.append(FLAG)
    return argv,final,saved


def replay_command(source,output):
    argv = previous.replay_command(source,output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    return argv


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode=='replay':
        replay(sys.argv[1:])
    elif mode=='native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output',sys.argv[2]]:
            raise ValueError('Keep the exact native observation output contract')
        native(Path(sys.argv[2]),sys.argv[boundary+1:])
    else:
        raise ValueError('Only declared native learning or frozen replay entry points')
