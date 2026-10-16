"""Train leg exploration without multiplying wheel exploration as well.

Only the entropy term changes. Actions, likelihoods, inference, trainable
parameters and checkpoint/Adam keys retain the ordinary ActorCritic contract.
"""
from functools import partial
import math
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic


def entropy_multipliers(leg_coefficient, wheel_coefficient=.001):
    for value in (leg_coefficient, wheel_coefficient):
        if type(value) not in (int, float) or not math.isfinite(value) or not .001 <= value <= .02:
            raise ValueError('Entropy coefficients must be finite in [.001, .02]')
    wheel = wheel_coefficient / leg_coefficient
    return [1., 1., wheel, 1., 1., wheel]


class JointPdExplorationActorCritic(ActorCritic):
    def __init__(self, *args, leg_entropy_coefficient=.001, wheel_entropy_coefficient=.001,
                 entropy_evidence=None, **kwargs):
        multipliers = entropy_multipliers(leg_entropy_coefficient, wheel_entropy_coefficient)
        super().__init__(*args, **kwargs)
        if self.actor[-1].out_features != 6:
            raise ValueError('Require the original six-action direct-joint policy')
        self.register_buffer('_entropy_multipliers', torch.tensor(multipliers), persistent=False)
        self._uniform_entropy = leg_entropy_coefficient == wheel_entropy_coefficient
        if entropy_evidence is not None:
            import json
            evidence = dict(policy_class=type(self).__name__,
                leg_entropy_coefficient=leg_entropy_coefficient,
                wheel_entropy_coefficient=wheel_entropy_coefficient,
                dimension_multipliers=self._entropy_multipliers.tolist(),
                modified_sampling_or_log_prob=False, modified_inference=False,
                additional_parameters=0, additional_checkpoint_keys=0,
                all_action_dimensions_trainable=True, policy_accepted=False)
            path = Path(entropy_evidence)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(evidence, indent=2)+'\n')
            temporary.replace(path)

    @property
    def entropy(self):
        entropy = self.distribution.entropy()
        if self._uniform_entropy:
            return entropy.sum(dim=-1)  # exact upstream reduction at equal weights
        return (entropy * self._entropy_multipliers).sum(dim=-1)


def register_runner_policy(leg_coefficient, evidence):
    import rsl_rl.runners.on_policy_runner as runner
    entropy_multipliers(leg_coefficient)
    runner.JointPdExplorationActorCritic = partial(
        JointPdExplorationActorCritic, leg_entropy_coefficient=leg_coefficient,
        wheel_entropy_coefficient=.001, entropy_evidence=evidence)
