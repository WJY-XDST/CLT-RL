"""Read the KL used by the installed ordinary-PPO adaptive scheduler.

This optimizer pre-hook reads the existing update frame after its learning
rate decision. It never replaces PPO.update, losses, backward, the scheduler,
optimizer arguments, observations or RNG. Reports cover the full minibatches,
unlike the separate cached32 whole-update diagnostic.
"""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from rsl_rl.algorithms import PPO


EXPECTED_PPO_SHA256 = 'deafc8c947eba4df3e91b393869426cdab8d7b71e05974c3734125d2331d7d1c'


class AdaptiveKLObserver:
    def __init__(self, algorithm):
        assert type(algorithm) is PPO and not algorithm.policy.is_recurrent
        assert algorithm.schedule == 'adaptive' and algorithm.desired_kl is not None
        assert not algorithm.is_multi_gpu and algorithm.rnd is None
        source = Path(inspect.getfile(PPO))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_PPO_SHA256
        self.algorithm = algorithm
        self.update_code = PPO.update.__code__
        self.rows = []
        self.completed_updates = 0
        self.handle = algorithm.optimizer.register_step_pre_hook(self._before_step)

    def _before_step(self, optimizer, args, kwargs):
        frame = sys._getframe(1)
        try:
            for _ in range(16):
                if frame is None or frame.f_code is self.update_code:
                    break
                frame = frame.f_back
            assert frame is not None and frame.f_code is self.update_code
            values = frame.f_locals
            assert values['self'] is self.algorithm
            with torch.no_grad():
                old_mu = values['old_mu_batch'].detach()
                old_sigma = values['old_sigma_batch'].detach()
                mu = values['mu_batch'].detach()
                sigma = values['sigma_batch'].detach()
                assert old_mu.shape == old_sigma.shape == mu.shape == sigma.shape
                assert mu.ndim == 2 and mu.shape[1] == 6
                # Preserve the installed scheduler's epsilon and operation
                # order. This is an observation, not an alternative scheduler.
                terms = (torch.log(sigma / old_sigma + 1.e-5)
                         + (torch.square(old_sigma) + torch.square(old_mu - mu))
                         / (2. * torch.square(sigma)) - .5)
                summed = terms.sum(-1)
                error = (summed - values['kl'].detach()).abs().max()
                mean_error = (summed.mean() - values['kl_mean'].detach()).abs()
                assert bool(torch.isfinite(terms).all())
                assert float(error) < 1.e-7 and float(mean_error) < 1.e-7
                means = terms.mean(0).cpu().tolist()
                kl = float(values['kl_mean'])
            rates = [group['lr'] for group in optimizer.param_groups]
            assert all(rate == self.algorithm.learning_rate for rate in rates)
            self.rows.append(dict(minibatch=len(self.rows) + 1,
                full_minibatch_examples=int(mu.shape[0]),
                scheduler_KL_mean=kl, scheduler_KL_mean_by_head=means,
                independent_formula_max_abs_error=float(error),
                independent_mean_max_abs_error=float(mean_error),
                actual_adaptive_LR_after_scheduler=rates,
                desired_kl=self.algorithm.desired_kl))
        finally:
            # Never retain an update frame or its minibatch tensors.
            del frame
        return None

    def finish_update(self, output=None):
        count = self.algorithm.num_learning_epochs * self.algorithm.num_mini_batches
        assert len(self.rows) == count
        self.completed_updates += 1
        result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
            update=self.completed_updates, full_original_PPO_minibatches=count,
            head_order=['LB', 'LL', 'LW', 'RB', 'RL', 'RW'],
            rows=self.rows, installed_PPO_source_sha256=EXPECTED_PPO_SHA256,
            scheduler_and_learning_rate_not_replaced=True,
            extra_loss_backward_or_optimizer_step=False,
            observations_actions_rewards_and_RNG_not_modified=True,
            scope='Full minibatch KL read from the actual adaptive-scheduler frame before each original Adam step.',
            policy_accepted=False)
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            path = output / f'update_{self.completed_updates:04d}.json'
            assert not path.exists()
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        self.rows = []
        return result

    def close(self):
        self.handle.remove()


def install_for_training_observer(observer_class):
    """Add recording to the existing observer lifecycle without editing it."""
    original_init, original_update = observer_class.__init__, observer_class._update

    def init(instance, algorithm, *args, **kwargs):
        original_init(instance, algorithm, *args, **kwargs)
        instance._original_scheduler_KL_observer = AdaptiveKLObserver(algorithm)
        instance.handles.append(instance._original_scheduler_KL_observer.handle)

    def update(instance, *args, **kwargs):
        result = original_update(instance, *args, **kwargs)
        path = (instance.output / 'original_adaptive_KL'
                if instance.output is not None else None)
        record = instance._original_scheduler_KL_observer.finish_update(path)
        assert record['update'] == len(instance.updates)
        return result

    observer_class.__init__, observer_class._update = init, update

    def restore():
        observer_class.__init__, observer_class._update = original_init, original_update

    return restore
