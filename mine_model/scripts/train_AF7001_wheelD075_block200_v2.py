"""Use the freshly qualified review dependency; keep prepared PPO code intact."""
import argparse
from pathlib import Path

import train_AF7001_wheelD075_block200 as original


SCRIPT = Path(__file__).resolve()
original.QUALIFICATION = original.RESULTS / 'joint_pd_AF7001_wheelD075_qualification_20261023/post200_complete_qualification_v2.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    proof = original.read(original.QUALIFICATION)
    assert proof['fresh_module_binding_preflight']['all_five_review_contexts_bound_to_selected_runtime']
    original.run(args.output, args.previous, args.unit)
