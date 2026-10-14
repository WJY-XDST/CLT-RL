"""Strict overrides and a minimal read-only view of saved configurations."""

from copy import deepcopy
import warnings


# These fields never affected observations, commands or rewards. Old paired
# env.yaml files remain immutable; only their in-memory compatibility view is
# normalized. Do not ignore arbitrary unknown names or rename active fields.
RETIRED_CONFIG_FIELDS = frozenset({
    'obs_scales.dof_acc', 'obs_scales.torque',
    'commands.num_commands', 'rewards.max_contact_force',
})
RETIRED_SAMPLING_FIELDS = frozenset({
    'transition_env_fraction', 'reverse_env_fraction', 'yaw_env_fraction',
})


def strip_retired_config_fields(config):
    """Remove inactive saved fields without changing the checkpoint's env.yaml.

    Old sampling fractions are inactive only in grouped training or fixed-speed
    replay. Reject the retired random sampler instead of silently replacing it.
    """
    result = deepcopy(config)
    for key in RETIRED_CONFIG_FIELDS:
        group, field = key.split('.')
        if isinstance(result.get(group), dict):
            result[group].pop(field, None)
    commands = result.get('commands', {})
    if RETIRED_SAMPLING_FIELDS.intersection(commands):
        limits = commands.get('ranges_lin_vel_x', ())
        fixed_speed = len(limits) == 2 and limits[0] == limits[1]
        if not commands.get('grouped_training', False) and not fixed_speed:
            raise ValueError('Saved config uses the retired non-grouped random sampler')
        for field in RETIRED_SAMPLING_FIELDS:
            commands.pop(field, None)
    return result


def apply_config_overrides(config, overrides):
    retired = sorted(RETIRED_CONFIG_FIELDS.intersection(overrides))
    if retired:
        warnings.warn('Ignoring retired no-op config fields: ' + ', '.join(retired),
                      UserWarning, stacklevel=2)
    for key,value in overrides.items():
        if key in RETIRED_CONFIG_FIELDS:
            continue
        node=config
        parts=key.split('.')
        for part in parts[:-1]:
            if ((isinstance(node, dict) and part not in node)
                    or (not isinstance(node, dict) and not hasattr(node, part))):
                raise ValueError(f'Unknown override: {key}')
            node=node[part] if isinstance(node,dict) else getattr(node,part)
        leaf=parts[-1]
        if isinstance(node,dict):
            if leaf not in node: raise ValueError(f'Unknown override: {key}')
            node[leaf]=value
        else:
            if not hasattr(node,leaf): raise ValueError(f'Unknown override: {key}')
            setattr(node,leaf,value)
