"""Original full evaluator with both paired feature and tire-physics metadata."""
from pathlib import Path
import runpy
import sys

from joint_pd_continuation import ROOT

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_analytic_tire_runtime as tires


if __name__ == '__main__':
    source = Path(sys.argv[sys.argv.index('--checkpoint') + 1]).resolve()
    tires.require_source(source)
    sys.argv = [sys.argv[0], *tires.arguments(sys.argv[1:])]
    runpy.run_path(str(ROOT / 'mine_model/scripts/evaluate_joint_pd_height_features.py'), run_name='__main__')
