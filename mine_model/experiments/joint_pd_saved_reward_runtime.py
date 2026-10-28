"""Use the existing scalar workflow with explicit changed-reward metadata.

Only provenance and entry-point binding differ from the generic workflow.
The unchanged dispatcher, PPO, state loader, observers and numerical readers
are reused. Reward values come from a pinned recipe, never operating cases.
"""
import ast
from pathlib import Path
import sys

import joint_pd_saved_scalar_runtime as implementation

SCRIPT = Path(__file__).resolve()
GENERIC_SCRIPT = Path(__file__).with_name('joint_pd_saved_scalar_runtime.py')
namespace = implementation.recipe.__globals__
old_profile = implementation.profile_for_count


def profile_for_count(count):
    result = old_profile(count)
    spec = implementation.recipe()
    assert spec['reward_changes'] and not spec.get('command_changes')
    result.update(saved_controller_and_reward_values_retained=False,
        saved_controller_values_retained=True,
        source_saved_rewards_unchanged_on_disk=True,
        uniform_reward_changes=spec['reward_changes'],
        uniform_scalar_base_sha256=implementation.sha(GENERIC_SCRIPT),
        reward_metadata_frontend_sha256=implementation.sha(SCRIPT))
    return result


namespace['SCRIPT'] = SCRIPT
implementation.SCRIPT = SCRIPT
implementation.profile_for_count = profile_for_count
sys.modules['joint_pd_saved_reward_runtime'] = implementation

if __name__ == '__main__':
    # Execute the original CLI branch unchanged, after binding this profile.
    dispatcher = ast.parse(GENERIC_SCRIPT.read_text()).body[-1]
    assert isinstance(dispatcher, ast.If)
    assert ast.unparse(dispatcher.test) == "__name__ == '__main__'"
    namespace['__name__'] = '__main__'
    exec(compile(ast.Module(body=[dispatcher], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
else:
    sys.modules[__name__] = implementation
