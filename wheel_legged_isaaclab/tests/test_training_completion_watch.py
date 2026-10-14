"""Mock-only shutdown eligibility tests; no simulator or real power action."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mine_model/scripts'))
from watch_training_completion import eligible


class TestCompletionWatch(unittest.TestCase):
    def setUp(self):
        self.request = {'armed': True, 'relative_error_target': .2, 'leg_control_mode': 'explicit_vmc'}
        self.state = {'state': 'accepted', 'child_pid': None, 'checkpoint': '/candidate/model_1999.pt',
                      'overrides': {'leg_control_mode': 'explicit_vmc'},
                      'acceptance_criteria': {'relative_error': .2}}
        self.completed = {'replays_passed': True, 'documentation_saved': True, 'checkpoint_sha256': 'fixture'}

    def test_packaged_vmc_result_may_enter_full_recheck(self):
        self.assertTrue(eligible(self.state, self.completed, self.request))

    def test_active_training_or_finalizing_cannot_poweroff(self):
        for mutation in ({'state': 'training_pilot'}, {'child_pid': 123}, {'error': 'gate failed'}):
            state = {**self.state, **mutation}
            self.assertFalse(eligible(state, self.completed, self.request))

    def test_implicit_or_looser_result_cannot_enter_recheck(self):
        for mode, target in (('implicit_joint_reference', .2), ('explicit_vmc', .3), ('explicit_vmc', float('nan'))):
            state = copy.deepcopy(self.state)
            state['overrides']['leg_control_mode'] = mode
            state['acceptance_criteria']['relative_error'] = target
            self.assertFalse(eligible(state, self.completed, self.request))

    def test_missing_delivery_and_cancelled_request_fail_closed(self):
        for mutation in ({'replays_passed': False}, {'documentation_saved': False}, {'checkpoint_sha256': ''}):
            self.assertFalse(eligible(self.state, {**self.completed, **mutation}, self.request))
        self.assertFalse(eligible(self.state, self.completed, {**self.request, 'armed': False}))


if __name__ == '__main__':
    unittest.main()
