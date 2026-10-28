"""Run every original6401 review against the explicit167+33 recovered ledger."""
import json
from pathlib import Path

import review_source6201_lin300_block200 as original
import verify_lin300_recovered167_33 as verifier
import recover_lin300_native33_20261020 as recovery

SCRIPT = Path(__file__).resolve()
QUALIFICATION = recovery.QUALIFICATION / 'reviews/post200_original_review_qualification.json'


def run():
    # Only ownership/qualification paths and the raw-segment metadata reader
    # change. Commands, seeds, numeric readers and failed None values stay.
    for module in (original, original._api, original._base):
        module.QUALIFICATION = QUALIFICATION
        module.final_check = verifier
    original._api._api = original._base
    return original.run()


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('actual_updates',
        'all_original_and_retained_reviews_complete', 'policy_accepted') if k in result}), flush=True)
