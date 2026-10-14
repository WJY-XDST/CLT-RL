"""Small, simulator-independent checks when resuming native RSL-RL training."""
import math
from pathlib import Path


def aggregate_episode_logs(records):
    """RSL's first-dictionary/mean-of-step-means logger loses sparse events.

    Environment episode fields now contain one element per ended episode.
    Concatenation weights every event equally; ordinary scalar diagnostics keep
    their per-step means. Missing groups remain absent rather than becoming NaN.
    """
    import torch
    result = {}
    for key in dict.fromkeys(key for record in records for key in record):
        values = [torch.as_tensor(record[key]).detach().flatten() for record in records if key in record]
        values = [value for value in values if value.numel()]
        if not values:
            continue
        values = torch.cat([value.to(device=values[0].device, dtype=torch.float32) for value in values])
        result[key] = values.mean()
        if key == 'Episode_Termination/numerical':
            # RSL prints ratios with four decimals; a rare failure can round to
            # zero. Preserve the integer count so the native guard still sees it.
            result['Episode_Count/numerical'] = values.sum()
        if key == 'Episode_Termination/unsafe_contact_or_orientation':
            group = 'all'
        elif key.startswith('Episode_Termination/') and key.endswith('_unsafe'):
            group = key.split('/', 1)[1][:-len('_unsafe')]
        else:
            continue
        result[f'Episode_Count/{group}_ended'] = values.new_tensor(values.numel())
        result[f'Episode_Count/{group}_unsafe'] = values.sum()
    return result


def install_episode_log_aggregation(runner):
    """Adapt logging in the project, without modifying third-party PPO code."""
    original_log = runner.log

    def log_events(locs, *args, **kwargs):
        records = aggregate_episode_logs(locs.get('ep_infos', []))
        # Do not replace the original list: the runner clears it after logging.
        return original_log(dict(locs, ep_infos=[records] if records else []), *args, **kwargs)

    runner.log = log_events


class CheckpointStopRequested(Exception):
    """A requested stop after a complete PPO update, not a native signal."""


def install_checkpoint_stop(runner, stop_file, log_dir):
    """Save actor, critic and Adam at the completed-update logging boundary.

    RSL-RL calls log after assigning current_learning_iteration. Do not signal
    native PhysX threads to implement an ordinary user/service stop.
    """
    if stop_file is None:
        return
    marker, destination = Path(stop_file), Path(log_dir)
    original_log = runner.log

    def log_and_check(locals_, *args, **kwargs):
        original_log(locals_, *args, **kwargs)
        if marker.exists():
            path = destination / f'model_{runner.current_learning_iteration}_interrupted.pt'
            runner.save(str(path))
            print(f'[INFO]: Cooperative stop checkpoint saved to {path}', flush=True)
            raise CheckpointStopRequested(str(path))

    runner.log = log_and_check


def restore_adaptive_learning_rate(algorithm):
    """Keep PPO's KL scheduler aligned with the loaded Adam parameter groups.

    RSL-RL loads optimizer state but its separate learning_rate attribute keeps
    the constructor default. Its next adaptive update otherwise overwrites the
    successfully restored optimizer learning rate.
    """
    rates = [float(group['lr']) for group in algorithm.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Loaded optimizer learning rate must be finite and positive')
    if any(not math.isclose(rate, rates[0], rel_tol=1.e-12, abs_tol=0.) for rate in rates):
        raise ValueError('Scalar PPO scheduler cannot restore unequal optimizer learning rates')
    algorithm.learning_rate = rates[0]
    return rates[0]
