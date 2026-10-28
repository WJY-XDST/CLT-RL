"""Unchanged full27 deterministic replay of the source801 drift-repair branch."""
from pathlib import Path
import sys

from joint_pd_continuation import ROOT
from evaluate_source200_stationary_height import evaluator_tree

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source801_drift50_runtime as runtime


if __name__ == '__main__':
    values = list(sys.argv[1:])
    flag = '--frozen_drift50_reward_transfer'
    transition = flag in values
    if transition:
        values.remove(flag)
    source = Path(values[values.index('--checkpoint') + 1]).resolve()
    runtime.require_source(source, transition=transition)
    sys.argv = [sys.argv[0], *runtime.tires.arguments(values)]
    tree, original = evaluator_tree()
    exec(compile(tree, str(original), 'exec'), dict(
        __name__='__main__', __file__=str(original), _observe_source200=runtime.observe_environment))
