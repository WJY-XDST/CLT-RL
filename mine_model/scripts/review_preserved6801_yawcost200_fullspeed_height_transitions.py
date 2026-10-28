"""Reuse the original signed-height reader and cached6801 source traces."""
import argparse
import json
from pathlib import Path

from joint_pd_continuation import sha
import plain27_sampler_metadata_adapter as adapter
import joint_pd_preserved6801_yawcost200_runtime as runtime
import qualify_preserved6801_yawcost200_transfer as qualifier
import review_preserved6801_yawcost200_block200 as post
import review_preserved6801_leg005_fullspeed_height_transitions_v2 as template

SCRIPT = Path(__file__).resolve()
QUALIFICATION = qualifier.OUTPUT / 'fullspeed_height/qualification.json'
OUTPUT = runtime.RESULTS / 'joint_pd_preserved6801_yawcost200_fullspeed_height_transitions_source6801_candidate7001_20261022'
AFTER_UNIT = 'clt-joint-pd-preserved6801-yawcost200-original-and-retained-reviews-20261022.service'
_api = adapter.private_module(template.SCRIPT, '_yawcost200_original_signed_height_review')
for name in ('SCRIPT', 'QUALIFICATION', 'OUTPUT', 'AFTER_UNIT', 'runtime', 'qualifier', 'post'):
    setattr(_api, name, globals()[name])
ADAPTER = adapter.retarget(_api, ('qualify',),
    strings={'training_launch_v2.json': 'training_launch.json'})


def qualify():
    result = _api.qualify()
    result['source_pins'][str(template.SCRIPT)] = sha(template.SCRIPT)
    result['source_pins'][str(adapter.SCRIPT)] = sha(adapter.SCRIPT)
    result.update(launch_basename_metadata_adapter=ADAPTER,
        original_signed_height_numerical_operations_preserved=True)
    _api.core.intact(result['source_pins'])
    QUALIFICATION.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


run = _api.run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result}))
