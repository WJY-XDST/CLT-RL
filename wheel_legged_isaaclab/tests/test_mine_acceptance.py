"""Do not promote collapsed episodes or commands that were never reached."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
from summarize_mine_trial import assess

class TestMineAcceptance(unittest.TestCase):
    def fixture(self,root,*,failure=False,unreached=False,collapsed=False,speed=0.,actual_speed=0.,unreached_yaw=False):
        manifest={'seconds_per_case':4.,'policy_dt':1.,'cases':[{'name':'stand','speed':.6 if unreached else speed,'yaw':.3 if unreached_yaw else 0.,'height':.3}],
                  'seeds':[11,12,13],'checkpoint':'fixture.pt','checkpoint_sha256':'fixture','checkpoint_iteration':5}
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'completed.json').write_text(json.dumps({'completed':True}))
        with (root/'trace.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['case_id','seed','episode','time_s','terminated','timeout','speed_m_s','speed_cmd_m_s','yaw_rate_rad_s','yaw_cmd_rad_s','height_m','height_cmd_m','left_length_m','left_length_ref_m','right_length_m','right_length_ref_m','left_angle_rad','left_angle_ref_rad','right_angle_rad','right_angle_ref_rad','pitch_rad','roll_rad','left_wheel_rad_s','right_wheel_rad_s','left_wheel_ref_rad_s','right_wheel_ref_rad_s'])
            w.writeheader()
            for seed in manifest['seeds']:
                for t in range(1,5):
                    row={key:0. for key in w.fieldnames}
                    row.update(seed=seed,time_s=t,height_m=.3,height_cmd_m=.3)
                    row.update(speed_cmd_m_s=0. if unreached else speed,speed_m_s=actual_speed)
                    if collapsed: row['height_m']=.14
                    for side in ('left','right'):row.update({side+'_length_m':.26,side+'_length_ref_m':.26})
                    if failure and seed==11:row.update(terminated=int(t==2),episode=int(t>2))
                    w.writerow(row)

    def test_safe_complete_replay_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            self.assertTrue(assess(root)['passed'])

    def test_failed_seed_cannot_be_hidden_by_clean_reset_samples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,failure=True)
            report=assess(root)
            self.assertFalse(report['passed'])
            self.assertEqual(report['survivors'],2)
            self.assertIsNone(report['conditions'][0]['trials'][0]['metrics'])

    def test_zero_error_to_unreached_ramp_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,unreached=True)
            self.assertFalse(assess(root)['passed'])

    def test_no_reset_collapsed_stance_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,collapsed=True)
            report=assess(root)
            self.assertEqual(report['survivors'],3)
            self.assertFalse(report['passed'])

    def test_twenty_five_percent_speed_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,speed=.6,actual_speed=.45)
            report=assess(root)
            self.assertEqual(report['survivors'],3)
            self.assertFalse(report['passed'])
            self.assertAlmostEqual(report['conditions'][0]['trials'][0]['limits']['speed_mae_m_s'],.12)

    def test_yaw_ramp_must_reach_requested_turn(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root,unreached_yaw=True)
            self.assertFalse(assess(root)['passed'])

if __name__=='__main__':unittest.main()
