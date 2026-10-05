"""A failed replay or incomplete delivery must never invoke poweroff."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import complete_mine_training as completion

class TestCompletionGate(unittest.TestCase):
    def full_replay_fixture(self, root, seeds):
        export=root/'export';export.mkdir()
        checkpoint=root/'model.pt';checkpoint.write_bytes(b'verified candidate')
        cases=[{'name':n} for n in ('stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6','spin_left','spin_right','height_low','height_high')]
        paths=[root/'round'/'replay',root/'round'/'heldout_replay',root/'round'/'export_replay']
        for path,seed in zip(paths,seeds):
            path.mkdir(parents=True)
            (path/'manifest.json').write_text(json.dumps({'cases':cases,'seeds':seed,'seconds_per_case':30}))
        (export/'acceptance.json').write_text(json.dumps({'checkpoint':str(checkpoint),'training_assessment':str(paths[0]/'assessment.json'),'heldout_assessment':str(paths[1]/'assessment.json')}))
        state=root/'status.json';state.write_text(json.dumps({'state':'accepted','export_directory':str(export)}))
        criteria=json.loads((Path(completion.__file__).resolve().parents[1]/'config/tracking_acceptance.json').read_text())
        return state,{'passed':True,'resets':0,'total_trials':27,'checkpoint_sha256':completion.sha(checkpoint),
                      'acceptance_criteria':criteria,'criterion':'Synthetic fixture; not a real-policy acceptance result'}

    def delivery_fixture(self, root):
        state,report=self.full_replay_fixture(root,[[43,44,45],[83,84,85],[83,84,85]])
        module=ast.parse(Path(completion.__file__).read_text())
        literals={node.targets[0].id:ast.literal_eval(node.value) for node in ast.walk(module)
                  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                  and node.targets[0].id in ('physical_fields','source_paths')}
        for node in ast.walk(module):
            if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='physical_fields':
                literals['physical_fields']+=ast.literal_eval(node.value)
        runtime={key:0 for key in literals['physical_fields']}
        asset=root/'robot.usd';asset.write_bytes(b'closed chain')
        runtime.update(leg_control_mode='implicit_joint_reference',leg_joint_stiffness=300,leg_joint_damping=3,
                       leg_effort_limit=10,wheel_damping=.5,sim={'dt':.005},robot={'spawn':{'usd_path':str(asset)}})
        for path in (root/'round').iterdir():
            manifest=json.loads((path/'manifest.json').read_text());manifest['asset_sha256']=completion.sha(asset)
            manifest['heading']='constant requested yaw rate; no added heading feedback'
            (path/'manifest.json').write_text(json.dumps(manifest))
            (path/'resolved_env.json').write_text(json.dumps(runtime))
        export=root/'export'
        for name in ('policy.pt','policy.onnx'):(export/name).write_bytes(name.encode())
        (export/'sim2sim_contract.json').write_text(json.dumps({'runtime_config':runtime,
            'export_sha256':{name:completion.sha(export/name) for name in ('policy.pt','policy.onnx')}}))
        fixture_root=root/'source'
        tutorial=fixture_root/'mine_model/docs/开链转闭链图文教程_v3';tutorial.parent.mkdir(parents=True)
        for suffix in ('.docx','.pdf'):tutorial.with_suffix(suffix).write_bytes(b'reviewed document')
        tutorial.with_suffix('.qa.json').write_text(json.dumps({'visually_verified':True,
            'sha256':{s:completion.sha(tutorial.with_suffix(s)) for s in ('.docx','.pdf')}}))
        for relative in literals['source_paths']:
            path=fixture_root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source snapshot')
        asset_copy=fixture_root/'wheel_legged_isaaclab/assets/robots/mine/robot.usd'
        asset_copy.parent.mkdir(parents=True);asset_copy.write_bytes(asset.read_bytes())
        (root/'params').mkdir();(root/'params/env.yaml').write_text('parameters')
        (root/'round/overrides.json').write_text('{}')
        report.update(criterion='fixture only',conditions=[{'name':'stand','trials':[{
            'metrics':{k:0 for k in ('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','leg_length_mae_m',
                                   'leg_angle_mae_rad','wheel_speed_mae_rad_s','body_p95_deg')},
            'limits':{k:1 for k in ('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','leg_length_mae_m',
                                   'leg_angle_mae_rad','wheel_speed_mae_rad_s','body_p95_deg')}}]}])
        return state,report,fixture_root

    def finalize_fixture(self, root, cancel=False, mutation=None):
        state,report,fixture_root=self.delivery_fixture(root)
        if cancel:(root/'CANCEL_SHUTDOWN').touch()
        if mutation:mutation(root)
        with patch.object(completion,'ROOT',fixture_root),patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'assess',return_value=report),patch.object(completion.subprocess,'run') as run:
            run.return_value.returncode=0
            # The training-record builder also calls git through the same
            # subprocess module. Its text outputs must be strings, not mocks.
            run.return_value.stdout=''
            run.return_value.stderr=''
            completion.main()
            return [call.args[0] for call in run.call_args_list if call.args[0][0]!='git']

    def test_complete_delivery_precedes_mocked_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);commands=self.finalize_fixture(root)
            self.assertTrue((root/'accepted_model_and_documents.zip').is_file())
            self.assertTrue(json.loads((root/'completion.json').read_text())['documentation_saved'])
            self.assertEqual(commands,[['/usr/bin/sync'],['/usr/bin/systemctl','--no-ask-password','poweroff']])

    def test_cancel_marker_saves_delivery_without_poweroff(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);commands=self.finalize_fixture(root,cancel=True)
            self.assertEqual(commands,[['/usr/bin/sync']])
            self.assertTrue(json.loads((root/'completion.json').read_text())['shutdown_cancelled'])

    def test_initialization_evidence_is_saved_and_tampering_blocks_shutdown(self):
        def add_event(root,tampered=False):
            manifest=root/'initialization.json';manifest.write_text('{"fraction":0.25}')
            checkpoint=root/'warm_start.pt';checkpoint.write_bytes(b'controlled initialization')
            statepath=root/'status.json';state=json.loads(statepath.read_text())
            state['interventions']=[{'checkpoint':str(checkpoint),
                'checkpoint_sha256':completion.sha(checkpoint),
                'initialization_manifest':str(manifest),'angle_initialization_manifest':str(manifest)}]
            statepath.write_text(json.dumps(state))
            if tampered:checkpoint.write_bytes(b'changed weights')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.finalize_fixture(root,mutation=add_event)
            for key in ('initialization_manifest','angle_initialization_manifest'):
                self.assertTrue((root/'final_delivery/optimization_history/intervention_000'/(key+'.json')).is_file())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'Initialization checkpoint differs'):
                self.finalize_fixture(Path(d),mutation=lambda r:add_event(r,True))

    def test_changed_export_asset_or_controller_blocks_shutdown(self):
        changes=[(lambda r:(r/'export/policy.onnx').write_bytes(b'tampered'),'Export changed'),
                 (lambda r:(r/'round/heldout_replay/manifest.json').write_text(
                    json.dumps({**json.loads((r/'round/heldout_replay/manifest.json').read_text()),'command_scale':.4})),'Diagnostic command amplitude'),
                 (lambda r:(r/'robot.usd').write_bytes(b'tampered'),'Robot asset differs'),
                 (lambda r:(r/'round/heldout_replay/resolved_env.json').write_text(
                    (r/'round/heldout_replay/resolved_env.json').read_text().replace('"leg_joint_stiffness": 300','"leg_joint_stiffness": 301')),'Controller differs')]
        for mutate,message in changes:
            with self.subTest(message=message),tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError,message):self.finalize_fixture(Path(d),mutation=mutate)

    def test_reused_reset_seeds_cannot_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            state,report=self.full_replay_fixture(Path(d),[[43,44,45],[43,84,85],[43,84,85]])
            with patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'assess',return_value=report), patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'seeds overlap'):
                    completion.main()
                run.assert_not_called()

    def test_changed_weights_cannot_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            state,report=self.full_replay_fixture(Path(d),[[43,44,45],[83,84,85],[83,84,85]])
            report['checkpoint_sha256']='old successful weights'
            with patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'assess',return_value=report), patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'Checkpoint differs'):
                    completion.main()
                run.assert_not_called()

    def test_protected_stop_asset_needs_cad_evidence_before_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            state,report,fixture_root=self.delivery_fixture(root)
            asset=root/'robot.usd'
            chain={str(asset):completion.sha(asset)}
            asset.with_suffix('.json').write_text(json.dumps({
                'sha256':chain,'confirmed_stop_pairs':[{'stop':'protected CAD'}]}))
            contract=root/'export/sim2sim_contract.json'
            value=json.loads(contract.read_text());value['asset_chain_sha256']=chain
            contract.write_text(json.dumps(value))
            for path in (root/'round').glob('*/manifest.json'):
                value=json.loads(path.read_text());value['asset_chain_sha256']=chain
                path.write_text(json.dumps(value))
            with patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'ROOT',fixture_root), patch.object(completion,'assess',return_value=report), patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'CAD stop contact evidence'):
                    completion.main()
                run.assert_not_called()

    def test_training_state_cannot_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'status.json'
            state.write_text(json.dumps({'state':'training_pilot'}))
            with patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'No accepted'):
                    completion.main()
                run.assert_not_called()

    def test_stale_accepted_label_does_not_bypass_failed_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);export=root/'export';export.mkdir()
            (export/'acceptance.json').write_text(json.dumps({'checkpoint':'model.pt','training_assessment':str(root/'replay/assessment.json'),'heldout_assessment':str(root/'heldout/assessment.json')}))
            state=root/'status.json'
            state.write_text(json.dumps({'state':'accepted','export_directory':str(export)}))
            with patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'assess',return_value={'passed':False,'resets':1}), patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'Replay acceptance failed'):
                    completion.main()
                run.assert_not_called()

if __name__=='__main__':unittest.main()
