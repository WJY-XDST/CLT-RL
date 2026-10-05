"""Explicit, reversible warm-start transformations of saved network parameters."""
import copy
import torch


def mirror_speed_command_initialization(checkpoint):
    """Reindex the learned +/- speed responses without changing the interface.

    Both networks receive the ordinary 27-D observation afterward. This is a
    new checkpoint initialization, not a runtime observation remapping. Adam's
    first moment follows the parameter reflection; its second moment is kept.
    Only checkpoints with the project's unnormalized feed-forward MLP contract
    are supported, so a normalizer cannot silently defeat the transformation.
    """
    result=copy.deepcopy(checkpoint)
    state=result['model_state_dict']
    if any('normalizer' in key for key in state):
        raise ValueError('Normalized checkpoint requires a separate transformation')
    keys=list(state)
    optimizer=result.get('optimizer_state_dict')
    ids=[i for g in optimizer['param_groups'] for i in g['params']] if optimizer else []
    if optimizer and len(ids)!=len(keys):
        raise ValueError('Optimizer parameter layout does not match saved network')
    for key in ('actor.0.weight','critic.0.weight'):
        weight=state[key]
        if weight.ndim!=2 or weight.shape[1]!=27 or not torch.isfinite(weight).all():
            raise ValueError('Expected finite 27-input actor and critic')
        weight[:,6].neg_()
        if optimizer:
            moments=optimizer['state'].get(ids[keys.index(key)],{})
            if 'exp_avg' in moments:
                if moments['exp_avg'].shape!=weight.shape:
                    raise ValueError('Optimizer first moment shape mismatch')
                moments['exp_avg'][:,6].neg_()
    return result


def center_leg_angle_initialization(checkpoint, fraction, nominal_action):
    """Shrink the deterministic angle means toward a verified standing mean.

    Retain exploration and other action means. Clear Adam history only for
    the edited final-layer rows, whose physical outputs have changed.
    """
    if not 0.<fraction<=1. or not -1.<=nominal_action<=1.:
        raise ValueError('Invalid angle initialization fraction or reference')
    result=copy.deepcopy(checkpoint);state=result['model_state_dict']
    keys=list(state);optimizer=result.get('optimizer_state_dict')
    ids=[i for g in optimizer['param_groups'] for i in g['params']] if optimizer else []
    if optimizer and len(ids)!=len(keys):raise ValueError('Optimizer layout mismatch')
    for key in ('actor.6.weight','actor.6.bias'):
        value=state[key]
        if value.shape[0]!=6:raise ValueError('Expected six action outputs')
        value[[0,3]]*=fraction
        if key.endswith('bias'):value[[0,3]]+=(1.-fraction)*nominal_action
        if optimizer:
            moments=optimizer['state'].get(ids[keys.index(key)],{})
            for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if name in moments:moments[name][[0,3]]=0.
    return result


def scale_actor_feedback_initialization(checkpoint, columns, fraction):
    """Reduce selected actor input gains for a controlled initialization trial.

    The observation contract and critic remain unchanged. Clear only edited
    actor-column Adam history; retain all other optimizer state and exploration.
    This does not prove stability and requires complete physical replay.
    """
    columns = list(columns)
    if (not 0. < fraction <= 1. or not columns or len(set(columns)) != len(columns)
            or any(not isinstance(i, int) or i < 0 or i >= 27 for i in columns)):
        raise ValueError('Invalid feedback initialization columns or fraction')
    result = copy.deepcopy(checkpoint)
    state = result['model_state_dict']
    if any('normalizer' in key for key in state):
        raise ValueError('Normalized checkpoint requires a separate transformation')
    key = 'actor.0.weight'
    weight = state[key]
    if weight.ndim != 2 or weight.shape[1] != 27 or not torch.isfinite(weight).all():
        raise ValueError('Expected finite 27-input actor')
    weight[:, columns] *= fraction
    optimizer = result.get('optimizer_state_dict')
    if optimizer:
        keys = list(state)
        ids = [i for g in optimizer['param_groups'] for i in g['params']]
        if len(ids) != len(keys):
            raise ValueError('Optimizer layout mismatch')
        moments = optimizer['state'].get(ids[keys.index(key)], {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in moments:
                if moments[name].shape != weight.shape:
                    raise ValueError('Optimizer moment shape mismatch')
                moments[name][:, columns] = 0.
    return result


def scale_wheel_difference_initialization(checkpoint, fraction):
    """Retain wheel mean and reduce the differential component of saved actor means.

    This is an initialization candidate, not an additional runtime controller.
    It changes turn response and must pass full fixed-command replay.
    """
    if not 0. < fraction <= 1.:
        raise ValueError('Invalid wheel differential fraction')
    result = copy.deepcopy(checkpoint)
    state = result['model_state_dict']
    keys = list(state)
    optimizer = result.get('optimizer_state_dict')
    ids = [i for g in optimizer['param_groups'] for i in g['params']] if optimizer else []
    if optimizer and len(ids) != len(keys):
        raise ValueError('Optimizer layout mismatch')
    for key in ('actor.6.weight', 'actor.6.bias'):
        value = state[key]
        if value.shape[0] != 6 or not torch.isfinite(value).all():
            raise ValueError('Expected finite six-action actor output')
        left, right = value[2].clone(), value[5].clone()
        mean, difference = (left + right) / 2., (left - right) / 2.
        value[2], value[5] = mean + fraction * difference, mean - fraction * difference
        if optimizer:
            moments = optimizer['state'].get(ids[keys.index(key)], {})
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if name in moments:
                    if moments[name].shape != value.shape:
                        raise ValueError('Optimizer moment shape mismatch')
                    moments[name][[2, 5]] = 0.
    return result
