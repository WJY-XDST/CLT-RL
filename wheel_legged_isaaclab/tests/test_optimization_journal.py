"""The report is append-only, recoverable and distinguishes learning from edits."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mine_model/scripts'))
from optimization_journal import sync_journal, WATCHED


class TestOptimizationJournal(unittest.TestCase):
    def fixture(self, root):
        output = root / 'mine_model/results/experiment'
        output.mkdir(parents=True)
        (output / 'status.json').write_text(json.dumps({'checkpoint': 'model_10.pt'}))
        source = root / WATCHED[0]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('gain = 1.0\n')
        return output, source

    def test_initial_baseline_and_restart_do_not_duplicate_or_rewrite_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, _ = self.fixture(root)
            report = sync_journal(output, root=root)
            initial = report.read_bytes()
            sync_journal(output, root=root)
            self.assertEqual(report.read_bytes(), initial)

    def test_removed_output_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'removed_ik_run'
            with self.assertRaises(FileNotFoundError):
                sync_journal(output, root=Path(d))
            self.assertFalse(output.exists())

    def test_retired_mode_is_rejected_before_report_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, _ = self.fixture(root)
            (output / 'status.json').write_text(json.dumps({
                'overrides': {'leg_control_mode': 'implicit_joint_reference'}}))
            with self.assertRaisesRegex(ValueError, 'Retired'):
                sync_journal(output, root=root)
            self.assertFalse((output / '持续优化报告.md').exists())
            self.assertFalse((output / 'optimization_journal').exists())

    def test_direct_joint_position_report_remains_supported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, _ = self.fixture(root)
            (output / 'status.json').write_text(json.dumps({
                'overrides': {'leg_control_mode': 'implicit_joint_position'}}))
            self.assertTrue(sync_journal(output, root=root).is_file())

    def test_source_change_appends_and_keeps_real_before_after_diff(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, source = self.fixture(root)
            report = sync_journal(output, root=root); initial = report.read_bytes()
            source.write_text('gain = 2.0\n')
            sync_journal(output, root=root)
            self.assertTrue(report.read_bytes().startswith(initial))
            patch = next((output / 'optimization_journal').glob('*.patch')).read_text()
            self.assertIn('-gain = 1.0', patch)
            self.assertIn('+gain = 2.0', patch)
            self.assertIn('正在运行的进程不会自动重载', report.read_text())

    def test_repeated_edit_after_revert_is_a_new_event(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, source = self.fixture(root)
            report = sync_journal(output, root=root)
            for value in (2, 1, 2):
                source.write_text(f'gain = {value}.0\n')
                sync_journal(output, root=root)
            self.assertEqual(report.read_text().count('### 源码修改：保存差异'), 3)

    def test_unchanged_round_is_only_continued_learning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, _ = self.fixture(root)
            log = root / 'mine_model/build/logs/experiment_service.log'
            log.parent.mkdir(parents=True)
            events = []
            for rid in (1, 2):
                events.extend([{'purpose': 'continue learning'},
                    {'child_log': str(output / f'round_{rid:03d}_optimization/training.log'),
                     'child_command': ['/project/rsl_rl/train.py', '--num_envs', '8192',
                                       'env.gain=1.0', '--target_iteration', str(rid * 200)]}])
            log.write_text('\n'.join(json.dumps(e) for e in events))
            report = sync_journal(output, root=root)
            self.assertIn('round_002_optimization：仅续训', report.read_text())
            self.assertIn('参数未变化：仅续训，不计为一次调参优化', report.read_text())

    def test_new_assessment_appends_results_without_altering_previous_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output, _ = self.fixture(root)
            report = sync_journal(output, root=root); initial = report.read_bytes()
            replay = output / 'round_001_optimization/replay'; replay.mkdir(parents=True)
            data = dict(checkpoint='model_10.pt', iteration=10, survivors=1, total_trials=1,
                        resets=0, passed=False, criterion='20 percent; zero command absolute tolerance',
                        conditions=[dict(name='stand', trials=[dict(seed=43, passed=False,
                        metrics={'yaw_mae_rad_s': .05}, limits={'yaw_mae_rad_s': .025})])])
            (replay / 'assessment.json').write_text(json.dumps(data))
            sync_journal(output, root=root)
            self.assertTrue(report.read_bytes().startswith(initial))
            self.assertIn('|stand|43|yaw_mae_rad_s|0.05|0.025|2.000|', report.read_text())
            content = report.read_bytes(); sync_journal(output, root=root)
            self.assertEqual(report.read_bytes(), content)

    def test_explicit_branch_launches_are_read_from_local_service_log(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); output,_=self.fixture(root)
            (output/'service.log').write_text(json.dumps({
                'purpose':'explicit VMC launch',
                'child_log':str(output/'round_000_pilot/training.log'),
                'child_command':['/project/scripts/rsl_rl/train.py','--num_envs','8192',
                                 'env.leg_control_mode="explicit_vmc"','--target_iteration','200']}))
            report=sync_journal(output,root=root)
            self.assertIn('round_000_pilot：参数调整/启动配置',report.read_text())
            self.assertIn('explicit_vmc',report.read_text())
            self.assertIn('8192',report.read_text())


if __name__ == '__main__':
    unittest.main()
