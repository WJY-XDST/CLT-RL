"""Read the actual declared first-rollout distribution without editing a model.

Two existing density/parity readers need the post-load std at that phase.
Their equations stay unchanged. Only a load of the exact original checkpoint
receives an in-memory std copy; later checkpoints keep their saved values.
"""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

from resume_std_pulse import digest as sha, equal


ROOT = Path(__file__).resolve().parents[2]
READERS = frozenset(('audit_joint_pd_policy_updates.py',
                     'audit_parallel8192_saved_action_limits_v2.py'))


def read(path):
    return json.loads(Path(path).read_text())


class DeclaredVectorDistribution:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.plan = read(self.directory / 'experiment_plan.json')
        self.native = self.directory / 'native_learning_signal'
        self.pulse = read(self.native / 'explicit_postload_exploration_runtime.json')
        self.source = Path(self.plan['source_checkpoint']).resolve()
        self.digest = sha(self.source)
        p = self.pulse
        assert p['applied'] and p['loaded_checkpoint_sha256'] == self.digest
        self.updates = self.plan['source_actual_ppo_updates']
        assert p['source_actual_updates'] == self.updates
        assert p['all17_Adam_step'] == self.updates * 20
        assert p['phase'] == 'after_complete_restore_and_base_future_floor_before_first_stochastic_rollout'
        assert p['only_std_tensor_intentionally_changed']
        assert p['all_other_policy_state_tensors_bitwise_preserved']
        assert p['all17_Adam_state_and_parameter_groups_bitwise_preserved']
        assert p['source_file_unchanged'] and p['RNG_unchanged']
        assert p['additional_observation_queries'] == p['additional_action_samples'] == p['optimizer_steps_executed'] == 0
        assert p['future_vector_floor_installed_after_declared_adjustment']
        self.before = torch.tensor(p['before_std'], dtype=torch.float32)
        self.after = torch.tensor(p['after_std'], dtype=torch.float32)
        minima = torch.tensor(p['declared_minimum_std_by_head'], dtype=torch.float32)
        assert torch.equal(self.after, torch.maximum(self.before, minima))
        assert torch.equal(self.after[[2, 5]], self.before[[2, 5]])
        assert p['actually_changed_heads'] == [0, 1, 3, 4]
        first = read(self.native / 'audit_update_0001.json')
        assert first['total_environments'] == self.plan['native_num_envs'] == 8192
        assert first['native_alignment_recorded']
        assert len(first['all_environment_alignment_max_errors']) == 9
        assert not any(first['all_environment_alignment_max_errors'].values())
        with np.load(self.native / 'rollout_0001.npz', allow_pickle=False) as arrays:
            assert arrays['sigma'].shape == (48, 32, 6)
            assert np.array_equal(arrays['sigma'], np.broadcast_to(self.after.numpy(), arrays['sigma'].shape))
        self.calls = 0

    @contextmanager
    def install(self):
        original = torch.load

        def load(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if isinstance(path, (str, Path)) and Path(path).resolve() == self.source:
                assert sha(self.source) == self.digest
                equal(self.before, value['model_state_dict']['std'].cpu(), 'saved_source_std')
                assert len(value['optimizer_state_dict']['state']) == 17
                assert {float(v['step']) for v in value['optimizer_state_dict']['state'].values()} == {self.updates * 20}
                before = copy.deepcopy(value)
                value['model_state_dict']['std'] = self.after.to(value['model_state_dict']['std'].device).clone()
                for name, item in before.items():
                    if name != 'model_state_dict':
                        equal(item, value[name], 'unchanged_source/' + name)
                for name, item in before['model_state_dict'].items():
                    if name != 'std':
                        equal(item, value['model_state_dict'][name], 'unchanged_mean_value/' + name)
                self.calls += 1
            return value

        torch.load = load
        try:
            yield
        finally:
            torch.load = original
            assert sha(self.source) == self.digest


def qualify(directory):
    directory = Path(directory).resolve()
    path = directory / 'declared_vector_distribution_CPU_verified.json'
    if path.exists():
        result = read(path)
        assert result['qualified']
        for item, digest in result['source_pins'].items():
            assert sha(item) == digest, item
        return result
    adapter = DeclaredVectorDistribution(directory)
    source = torch.load(adapter.source, map_location='cpu', weights_only=True)
    final = read(directory / 'final_checkpoint_verification_readonly.json')
    checkpoint = Path(final['checkpoint']).resolve()
    assert final['verified'] and final['all50_native_updates_verified']
    assert final['checkpoint_sha256'] == sha(checkpoint)
    assert final['actual_ppo_updates'] == adapter.updates + 50
    candidate = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rng = torch.random.get_rng_state().clone()
    with adapter.install():
        reconstructed = torch.load(adapter.source, map_location='cpu', weights_only=True)
        later = torch.load(checkpoint, map_location='cpu', weights_only=True)
    equal(candidate, later, 'later_checkpoint_bitwise_unmodified')
    equal(source['optimizer_state_dict'], reconstructed['optimizer_state_dict'], 'source17_Adam_unmodified')
    assert torch.equal(reconstructed['model_state_dict']['std'], adapter.after)
    assert adapter.calls == 1 and torch.equal(rng, torch.random.get_rng_state())
    result = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        exact_origin_only_in_memory_std_reconstruction=True,
        native_first_rollout_sigma_and_all9_alignment_checks_verified=True,
        source_file_and_later_checkpoints_unmodified=True,
        original_numeric_reader_equations_unchanged=True,
        before_std=adapter.before.tolist(), actual_pre_rollout_std=adapter.after.tolist(),
        initial_wheel_std_unchanged=True, CPU_RNG_unchanged=True,
        optimizer_steps_executed=0, simulator_started=False, policy_accepted=False,
        source_pins={str(p): sha(p) for p in (Path(__file__), adapter.source, checkpoint,
            directory / 'experiment_plan.json', directory / 'final_checkpoint_verification_readonly.json',
            adapter.native / 'explicit_postload_exploration_runtime.json',
            adapter.native / 'audit_update_0001.json', adapter.native / 'rollout_0001.npz')})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def run(reader, directory, arguments):
    reader = Path(reader).resolve()
    assert reader.parent == ROOT / 'mine_model/scripts' and reader.name in READERS
    qualify(directory)
    adapter = DeclaredVectorDistribution(directory)
    original_argv = sys.argv
    try:
        sys.argv = [str(reader), str(directory), *arguments]
        with adapter.install():
            runpy.run_path(str(reader), run_name='__main__')
    finally:
        sys.argv = original_argv
    result = dict(verified=True, reader=str(reader), original_reader_sha256=sha(reader),
        original_numeric_reader_unchanged=True, reconstructed_origin_loads=adapter.calls,
        in_memory_std_override_only=True, actual_pre_rollout_std=adapter.after.tolist(),
        source_checkpoint_sha256=adapter.digest,
        source_file_and_all_later_checkpoints_unchanged=True,
        no_native_checkpoint_created=True, optimizer_steps_executed=0, simulator_started=False,
        helper_sha256=sha(__file__), policy_accepted=False)
    destination = Path(directory) / ('declared_preupdate_phase_' + reader.stem + '_verified.json')
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reader-script', dest='reader', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    run(args.reader, args.directory.resolve(), remaining)
