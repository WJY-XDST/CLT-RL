"""Same feature PPO experiment, with the qualified post-replay actor reader."""
import argparse
from pathlib import Path

from auto_train_mine import write_json
import continue_joint_pd_height_features as feature
import diagnose_height_feature_feedback as feedback

ROOT = feature.ROOT
QUALIFIED = ROOT / 'mine_model/results/joint_pd_optimization_20261017_030235_497149_stationary100/feature_feedback_reader_qualification.json'


def prepare(*args, **kwargs):
    plan, expected = feature.prepare(*args, **kwargs)
    for path in (Path(__file__).resolve(), Path(feedback.__file__).resolve(), QUALIFIED,
                 ROOT / 'mine_model/scripts/diagnose_height_feature_response.py',
                 ROOT / 'mine_model/scripts/diagnose_policy_feedback.py'):
        plan['source_files'][str(path)] = feature.core.sha(path)
    plan['post_replay_reader_revision'] = 'Feature-aware actor forward; original diagnostic metrics unchanged'
    write_json(Path(args[1]) / 'experiment_plan.json', plan)
    return plan, expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--environment-revision', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    feature.install()
    feature.framework.prepare = prepare
    feature.core.block.diagnose_joint_pd = feedback.diagnose_joint_pd
    feature.framework.run(args.previous_coordinator, args.coordinator, args.unit, args.environment_revision)
