"""Weighted targeted command grid for the retained ordinary-PPO sampler.

Only the three categorical grid draws change. The original 30% selection,
role masks, fixed standing IDs, bounds, speed-slew buffer and fixed replay
bypass stay identical. Both yaw signs retain equal probability. Every bin
remains possible, and the other 70% of eligible requests stay continuous.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch

import targeted_turn_command_sampler as original


SCRIPT = Path(__file__).resolve()
CONFIG_KEY = original.CONFIG_KEY
WEIGHTS = {
    "speed": (.05, .10, .10, .25, .50),
    "yaw": (.60, .30, .10),
    "height": (.45, .10, .45),
}


def weighted_draw(env, name, size):
    probabilities = env._commands.new_tensor(WEIGHTS[name])
    return torch.multinomial(probabilities, size, replacement=True)


def build_apply():
    tree = ast.parse(inspect.getsource(original.apply_after_original))
    before, edits = ast.dump(tree), []
    names = {"speed_bins": "speed", "yaw_bins": "yaw", "height_bins": "height"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in names:
            assert isinstance(node.slice, ast.Call) and ast.unparse(node.slice.func) == "torch.randint"
            edits.append((node, node.slice))
            node.slice = ast.parse("_weighted_draw(env, " + repr(names[node.value.id]) + ", size)", mode="eval").body
    assert len(edits) == 3
    modified = copy.deepcopy(tree)
    for node, value in edits:
        node.slice = value
    assert ast.dump(tree) == before
    namespace = dict(original.__dict__, _weighted_draw=weighted_draw)
    exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), "exec"), namespace)
    evidence = dict(original_source=str(Path(original.__file__).resolve()),
                    original_function_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
                    original_AST_restored_identical=True, replaced_categorical_draws=3,
                    only_joint_grid_probabilities_changed=True,
                    original_role_masks_ranges_standing_and_slew_preserved=True,
                    targeted_fraction_unchanged=.3, yaw_sign_probability=.5,
                    weights={name: list(value) for name, value in WEIGHTS.items()},
                    targeted_random_draws_intentionally_changed=True,
                    no_random_draw_on_fixed_or_zero_fraction_replay=True)
    return namespace["apply_after_original"], evidence


apply_after_original, AST_EVIDENCE = build_apply()


def install(env):
    """Replace the existing grid wrapper on one environment, without resampling."""
    if getattr(env, "_weighted_joint_grid_installed", False):
        return
    if not getattr(env, "_targeted_turn_sampler_installed", False):
        raise ValueError("Install the retained sampler before the weighted grid wrapper")
    if float(getattr(env.cfg.commands, CONFIG_KEY)) != .3:
        raise ValueError("Keep the original targeted fraction exactly0.3")
    base = env._targeted_turn_sampler_original_resample

    def resample(self, env_ids):
        base(env_ids)
        selected = apply_after_original(self, env_ids)
        self._targeted_turn_sampler_calls += 1
        self._targeted_turn_sampler_selected += selected.numel()

    env._resample_commands_for = MethodType(resample, env)
    env._weighted_joint_grid_installed = True


def expected_probability_changes():
    """Balanced global resampling probabilities, including unchanged role masks."""
    eligible_fraction = .75 * 3. / 5.
    role_fraction, sign_probability, selected_fraction = 1. / 3., .5, .3
    uniform = eligible_fraction * role_fraction * selected_fraction * sign_probability / (5. * 3. * 3.)
    focused = eligible_fraction * role_fraction * selected_fraction * sign_probability * WEIGHTS["speed"][-1] * WEIGHTS["yaw"][0] * WEIGHTS["height"][0]
    return dict(original_worst_reverse08_left0075_low_probability_per_all_id=uniform,
                weighted_worst_reverse08_left0075_low_probability_per_all_id=focused,
                worst_case_requested_target_probability_multiplier=focused / uniform,
                assumed_balanced_global_roles=True,
                not_a_measurement_of_actual_slew_settled_training_exposure=True)
