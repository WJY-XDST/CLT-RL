"""Pair cached7001 signed-height traces with unchanged7201 native trials."""
import argparse
import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import plain27_sampler_metadata_adapter as adapter
import source7001_yawcost200_unchanged_context as context
import qualify_retained_plain27_source as cpu_qualifier
import review_source7001_yawcost200_unchanged200 as post
import review_preserved6801_yawcost200_fullspeed_height_transitions as previous
import review_preserved6801_leg005_fullspeed_height_transitions_v2 as template

SCRIPT = Path(__file__).resolve()
QUALIFICATION = context.QUALIFICATION / 'fullspeed_height/qualification.json'
OUTPUT = context.EXTRA
AFTER_UNIT = 'clt-joint-pd-source7201-yawcost200-unchanged-original-reviews-20261022.service'
runtime = context.api_runtime()
qualifier = SimpleNamespace(OUTPUT=context.QUALIFICATION, SCRIPT=cpu_qualifier.SCRIPT)
_api = adapter.private_module(template.SCRIPT, '_source7001_unchanged_cached_signed_height')
for name in ('SCRIPT', 'QUALIFICATION', 'OUTPUT', 'AFTER_UNIT', 'runtime', 'qualifier', 'post', 'previous'):
    setattr(_api, name, globals()[name])
_api.SOURCE = previous.OUTPUT / 'candidate7001'
labels = {'training_launch_v2.json': 'training_launch.json'}
for function in (template.qualify, template.run):
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = re.sub(r'(source|candidate)(6801|7001)',
                lambda m: m.group(1) + {'6801': '7001', '7001': '7201'}[m.group(2)], node.value)
            if value != node.value:
                labels[node.value] = value
ADAPTER = adapter.retarget(_api, ('qualify', 'run'),
    integers={'*': {6801: 7001, 7001: 7201, 140020: 144020}},
    exact_strings=labels,
    keyword_names={'source6801': 'source7001', 'candidate7001': 'candidate7201'})
qualify, run = _api.qualify, _api.run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result})
