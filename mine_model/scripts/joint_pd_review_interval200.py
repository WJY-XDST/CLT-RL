"""Change only immutable observer review counters from 100 to 200 updates.

The historical 100-update sources and their checkpoint hashes stay intact.
No rollout, policy, optimizer, reward or control implementation is rewritten.
"""
import ast
from collections import Counter
import inspect
from pathlib import Path
import textwrap

from joint_pd_continuation import sha

UPDATES = 200
ADAM_STEPS_PER_UPDATE = 20


def _replace_interval(module, name):
    function = getattr(module, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    replacements = {100: UPDATES, 101: UPDATES + 1, 2000: UPDATES * ADAM_STEPS_PER_UPDATE}
    changed = Counter()

    class Interval(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in replacements:
                changed[str(node.value)] += 1
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            if isinstance(node.value, str) and '100' in node.value:
                changed['message'] += 1
                return ast.copy_location(ast.Constant(node.value.replace('100', str(UPDATES))), node)
            return node

    modified = Interval().visit(tree)
    if not changed['100']:
        raise ValueError('Original observer interval no longer matches: ' + name)
    exec(compile(ast.fix_missing_locations(modified), inspect.getsourcefile(function), 'exec'), module.__dict__)
    return dict(function=name, source=inspect.getsourcefile(function),
                source_sha256=sha(inspect.getsourcefile(function)), replacements=dict(changed))


def install_observer(module):
    return dict(actual_updates_per_review=UPDATES, optimizer_minibatches=UPDATES * ADAM_STEPS_PER_UPDATE,
                adapter=str(Path(__file__).resolve()), adapter_sha256=sha(__file__),
                only_counter_and_message_literals_changed=True,
                functions=[_replace_interval(module, name) for name in ('continuation_source', 'run')])


def install_finalizer(module):
    return dict(actual_updates_per_review=UPDATES, optimizer_minibatches=UPDATES * ADAM_STEPS_PER_UPDATE,
                only_counter_and_message_literals_changed=True,
                functions=[_replace_interval(module, name)
                           for name in ('summarize_observations', 'finalize_observations')])
