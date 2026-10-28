"""Explicit, process-local registration and full-state feature migration.

Only the new experimental entry points import this module. The original
trainer/evaluator, original checkpoint and ordinary PPO update stay intact.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch

from joint_pd_height_features_candidate import (
    feature_model_from_config, migrate_saved_state, register_runner_policy,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def saved_feature_model(saved):
    top = saved.get('joint_pd_height_features')
    infos = saved.get('infos')
    nested = infos.get('joint_pd_height_features') if isinstance(infos, dict) else None
    if top is not None and nested is not None and top != nested:
        raise ValueError('Conflicting checkpoint height feature definitions')
    return top if top is not None else nested


def require_paired_feature_model(saved, cfg, hip_z):
    model = saved_feature_model(saved)
    if model is None or model != feature_model_from_config(cfg, hip_z):
        raise ValueError('Missing or mismatched encoder/IMU feature metadata')
    if any(saved['model_state_dict'][name].shape[1] != 29
           for name in ('actor.0.weight', 'critic.0.weight')):
        raise ValueError('Feature checkpoint does not have its declared first-layer width')
    return model


def install_training(model, observation_output):
    """Use upstream load/save, with an explicit derived migration artifact."""
    import rsl_rl.runners as runners
    upstream = runners.OnPolicyRunner
    register_runner_policy(model)
    directory = Path(observation_output)

    class HeightFeatureRunner(upstream):
        def load(self, path, load_optimizer=True, map_location=None):
            if not load_optimizer:
                raise ValueError('Feature trials must retain the complete optimizer')
            source = Path(path).resolve()
            source_hash = sha(source)
            saved = torch.load(source, map_location='cpu', weights_only=True)
            width = saved['model_state_dict']['actor.0.weight'].shape[1]
            names = [name for name, _ in self.alg.policy.named_parameters()]
            if width == 27:
                if saved_feature_model(saved) is not None:
                    raise ValueError('A legacy checkpoint must not declare active feature columns')
                migrated = migrate_saved_state(saved, names, model)
                migration = directory / 'feature_migration'
                migration.mkdir(parents=True, exist_ok=False)
                derived = migration / source.name
                torch.save(migrated, derived)
                loaded = super().load(str(derived), load_optimizer=True, map_location=map_location)
                migrated_once = True
            elif width == 29:
                if saved_feature_model(saved) != model:
                    raise ValueError('Continue only the identical retained feature architecture')
                loaded = super().load(str(source), load_optimizer=True, map_location=map_location)
                derived, migrated_once = source, False
            else:
                raise ValueError('Unreviewed checkpoint first-layer width')
            parameters = list(self.alg.policy.named_parameters())
            if [name for name, _ in parameters] != names or len(parameters) != 17:
                raise ValueError('All original policy parameters must remain trainable')
            if len(self.alg.optimizer.state) != 17:
                raise ValueError('All 17 Adam states must be restored')
            expected_step = {float(value['step']) for value in saved['optimizer_state_dict']['state'].values()}
            loaded_step = {float(value['step']) for value in self.alg.optimizer.state.values()}
            if expected_step != loaded_step or len(expected_step) != 1:
                raise ValueError('Adam clock changed during loading')
            identifiers = [i for group in saved['optimizer_state_dict']['param_groups'] for i in group['params']]
            for (name, parameter), identifier in zip(parameters, identifiers):
                original = saved['model_state_dict'][name]
                current = parameter.detach().cpu()
                if name in ('actor.0.weight', 'critic.0.weight') and width == 27:
                    if not torch.equal(current[:, :27], original) or torch.count_nonzero(current[:, 27:]):
                        raise ValueError('Source parameter columns changed during migration')
                elif not torch.equal(current, original):
                    raise ValueError('Source parameter changed: ' + name)
                old_moments = saved['optimizer_state_dict']['state'][identifier]
                new_moments = self.alg.optimizer.state[parameter]
                if set(old_moments) != set(new_moments):
                    raise ValueError('Adam moment keys changed: ' + name)
                for key, moment in old_moments.items():
                    restored = new_moments[key]
                    if torch.is_tensor(moment):
                        restored = restored.detach().cpu()
                        if width == 27 and name in ('actor.0.weight', 'critic.0.weight') and moment.ndim:
                            if not torch.equal(restored[:, :27], moment) or torch.count_nonzero(restored[:, 27:]):
                                raise ValueError('Original Adam columns changed: ' + name)
                        elif not torch.equal(restored, moment):
                            raise ValueError('Adam state changed: ' + name + '/' + key)
                    elif restored != moment:
                        raise ValueError('Non-tensor Adam state changed: ' + name + '/' + key)
            if ([g['lr'] for g in self.alg.optimizer.param_groups]
                    != [g['lr'] for g in saved['optimizer_state_dict']['param_groups']]):
                raise ValueError('Optimizer learning rate changed during migration')
            if sha(source) != source_hash:
                raise ValueError('The retained source checkpoint was overwritten')
            write(directory / 'feature_runtime_load.json', dict(
                source_checkpoint=str(source), source_sha256=source_hash,
                loaded_derived_checkpoint=str(derived), loaded_derived_sha256=sha(derived),
                migrated_once=migrated_once, original_parameters_preserved=True,
                all17_Adam_restored=True, Adam_step=next(iter(loaded_step)),
                optimizer_LR=[group['lr'] for group in self.alg.optimizer.param_groups],
                feature_model=model, source_unchanged=True, optimizer_reset=False,
                added_loss=False, policy_accepted=False))
            self._feature_source_checkpoint = str(source)
            self._feature_source_sha256 = source_hash
            return loaded

        def save(self, path, infos=None):
            metadata = copy.deepcopy(infos) if isinstance(infos, dict) else {'upstream_infos': infos}
            if metadata.get('joint_pd_height_features', model) != model:
                raise ValueError('Never silently change a retained feature definition')
            metadata.update(joint_pd_height_features=copy.deepcopy(model),
                height_feature_source_checkpoint=getattr(self, '_feature_source_checkpoint', None),
                height_feature_source_sha256=getattr(self, '_feature_source_sha256', None),
                external_observation_size=27, external_action_size=6,
                feature_architecture='encoder_imu_support_height_error_v1')
            super().save(path, infos=metadata)

    runners.OnPolicyRunner = HeightFeatureRunner
    return upstream, HeightFeatureRunner
