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
import numpy as np


class TestSustainedAcceptance(unittest.TestCase):
    def replay(self, root, early_drift=False, missing=False, exact_limit=False):
        from summarize_mine_trial import assess_stability
        case=dict(name='stand',speed=0.,yaw=0.,height=.3)
        manifest=dict(checkpoint='fixture.pt',checkpoint_sha256='fixture',checkpoint_iteration=1,
                      cases=[case],seeds=[83],seconds_per_case=10.,policy_dt=.02)
        fields=['case_id','seed','episode','time_s','terminated','timeout','speed_m_s','speed_cmd_m_s',
                'yaw_rate_rad_s','yaw_cmd_rad_s','height_m','height_cmd_m','pitch_rad','roll_rad']
        for s in ('left','right'):
            fields += [s+'_length_m',s+'_length_ref_m',s+'_angle_rad',s+'_angle_ref_rad',s+'_wheel_rad_s',s+'_wheel_ref_rad_s']
        data=np.zeros(500,dtype=[(name,float) for name in fields])
        data['seed']=83;data['time_s']=np.arange(1,501)*.02
        data['height_m']=data['height_cmd_m']=.3
        for s in ('left','right'):data[s+'_length_m']=data[s+'_length_ref_m']=.27
        if early_drift:data['speed_m_s'][(data['time_s']>=4)&(data['time_s']<7)]=.1
        if exact_limit:data['speed_m_s']=.025
        if missing:data['time_s'][250]+=.003
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'completed.json').write_text(json.dumps(dict(completed=True,samples=500,resets=0)))
        np.savetxt(root/'trace.csv',data,delimiter=',',header=','.join(fields),comments='',fmt='%.10f')
        return assess_stability(root,6.)

    def test_all_steady_windows_are_required_not_only_a_good_final_tail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=self.replay(root,early_drift=True)
            self.assertTrue(json.loads((root/'assessment_final_gate.json').read_text())['passed'])
            self.assertFalse(report['passed'])
            self.assertGreater(report['conditions'][0]['trials'][0]['metrics']['speed_mae_m_s'],.025)

    def test_clean_full_windows_pass_without_relaxing_limits(self):
        with tempfile.TemporaryDirectory() as d:
            report=self.replay(Path(d))
            self.assertTrue(report['passed'])
            self.assertEqual(report['stability']['windows_per_trial'],4)

    def test_equal_limit_is_not_strictly_below_limit(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(self.replay(Path(d),exact_limit=True)['passed'])

    def test_inconsistent_time_cannot_count_as_stable(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'time sequence'):self.replay(Path(d),missing=True)

    def test_missing_extended_evidence_can_never_power_off(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'status.json';state.write_text(json.dumps(dict(state='training')))
            with patch.object(completion.subprocess,'run') as run:
                with self.assertRaises(RuntimeError):completion.finish_joint_pd(state,shutdown=True)
                run.assert_not_called()


class TestJointSavedDelivery(unittest.TestCase):
    """Synthetic gate fixtures; all poweroff calls are mocked, never executed."""
    def fixture(self, root):
        import torch
        import copy
        project=root/'project'
        output=project/'mine_model/results/joint_pd_optimization_fixture';output.mkdir(parents=True)
        run=project/'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/fixture';run.mkdir(parents=True)
        checkpoint=run/'model_1.pt'
        torch.save(dict(model_state_dict={'actor.weight':torch.zeros(1)},
                        optimizer_state_dict={'param_groups':[{'lr':.001}]},iter=1),checkpoint)
        asset=project/'robot.usd';asset.write_bytes(b'synthetic asset, not a real robot')
        source=project/'controller.py';source.write_text('# synthetic controller snapshot\n')
        module=ast.parse(Path(completion.__file__).read_text())
        physical=next(ast.literal_eval(n.value) for n in ast.walk(module)
                      if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='physical')
        runtime={k:0 for k in physical}
        runtime.update(leg_control_mode='implicit_joint_position',robot={'spawn':{'usd_path':str(asset)}},
                       sim={'dt':.005},commands={'ranges_height':[.28,.32]},rewards={'base_height_target':.3})
        params=run/'params';params.mkdir()
        for name in ('env.yaml','agent.yaml','control_contract.yaml','precision_mode.yaml'):
            (params/name).write_text('synthetic params\n')
        criteria=completion.acceptance_criteria()
        modes={'stand':(0.,0.,.3),'forward_0p3':(.3,0.,.3),'forward_0p6':(.6,0.,.3),
               'reverse_0p3':(-.3,0.,.3),'reverse_0p6':(-.6,0.,.3),'spin_left':(0.,.3,.3),
               'spin_right':(0.,-.3,.3),'height_low':(0.,0.,.28),'height_high':(0.,0.,.32)}
        cases=[dict(name=n,speed=v[0],yaw=v[1],height=v[2]) for n,v in modes.items()]
        directories=[output/n for n in ('training','heldout','extended')]
        chain={str(asset):completion.sha(asset)}
        for directory,seeds,seconds in zip(directories,([83,84,85],[183,184,185],[283,284,285]),(20,30,60)):
            directory.mkdir()
            manifest=dict(cases=cases,seeds=seeds,seconds_per_case=seconds,command_scale=1.,
                          checkpoint_sha256=completion.sha(checkpoint),asset_chain_sha256=chain,
                          heading='constant requested yaw rate; no added heading feedback')
            (directory/'manifest.json').write_text(json.dumps(manifest))
            (directory/'resolved_env.json').write_text(json.dumps(runtime))
            for name in ('completed.json','assessment_final_gate.json','assessment_stability.json'):
                (directory/name).write_text('{}')
        state=dict(state='simulation_accepted_pending_sim2sim_and_randomization',
                   best_checkpoint=str(checkpoint),acceptance_criteria=criteria,
                   source_files={str(source):completion.sha(source)})
        for key,directory in zip(('best_assessment','heldout_assessment','extended_assessment'),directories):
            state[key]=str(directory/'assessment.json')
        path=output/'status.json';path.write_text(json.dumps(state))
        report=dict(passed=True,resets=0,survivors=27,total_trials=27,acceptance_criteria=criteria,
                    checkpoint_sha256=completion.sha(checkpoint))
        return project,path,runtime,chain,report

    def finish(self, root, mutate=None):
        project,state,runtime,chain,report=self.fixture(root)
        if mutate:mutate(state.parent,report)
        with patch.object(completion,'ROOT',project),patch.object(completion,'assess',return_value=report),\
             patch.object(completion,'assess_stability',return_value={'passed':True}),\
             patch.object(completion,'verify_asset_chain',return_value=(chain,{})),\
             patch('wheel_legged_gym_isaaclab.robot_model.validate_checkpoint_robot',return_value=runtime),\
             patch.object(completion.subprocess,'run') as run:
            result=completion.finish_joint_pd(state,shutdown=True)
            self.assertTrue((state.parent/'final_delivery/model.pt').is_file())
            self.assertTrue((state.parent/'final_delivery/params/agent.yaml').is_file())
            return result,[c.args[0] for c in run.call_args_list]

    def test_verified_model_and_optimizer_are_saved_before_mocked_poweroff(self):
        with tempfile.TemporaryDirectory() as d:
            result,commands=self.finish(Path(d))
            self.assertTrue(result['model_saved'] and result['optimizer_saved'])
            self.assertEqual(commands,[['/usr/bin/sync'],['/usr/bin/systemctl','--no-ask-password','poweroff']])

    def test_cancel_preserves_saved_model_without_mocked_poweroff(self):
        with tempfile.TemporaryDirectory() as d:
            result,commands=self.finish(Path(d),lambda output,report:(output/'CANCEL_SHUTDOWN').touch())
            self.assertTrue(result['shutdown_cancelled'])
            self.assertEqual(commands,[['/usr/bin/sync']])

    def test_cancel_arriving_during_sync_prevents_poweroff(self):
        with tempfile.TemporaryDirectory() as d:
            project,state,runtime,chain,report=self.fixture(Path(d))
            def during_sync(command,**kwargs):
                self.assertEqual(command,['/usr/bin/sync'])
                (state.parent/'CANCEL_SHUTDOWN').touch()
            with patch.object(completion,'ROOT',project),patch.object(completion,'assess',return_value=report),\
                 patch.object(completion,'assess_stability',return_value={'passed':True}),\
                 patch.object(completion,'verify_asset_chain',return_value=(chain,{})),\
                 patch('wheel_legged_gym_isaaclab.robot_model.validate_checkpoint_robot',return_value=runtime),\
                 patch.object(completion.subprocess,'run',side_effect=during_sync) as run:
                result=completion.finish_joint_pd(state,True)
                self.assertTrue(result['shutdown_cancelled'])
                self.assertFalse(result['shutdown_requested'])
                self.assertEqual(run.call_count,1)

    def test_nonfinite_optimizer_refuses_save_and_poweroff(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            project,state,runtime,chain,report=self.fixture(Path(d))
            checkpoint=Path(json.loads(state.read_text())['best_checkpoint'])
            saved=torch.load(checkpoint,weights_only=True)
            saved['optimizer_state_dict']['state']={0:{'exp_avg':torch.tensor([float('nan')])}}
            torch.save(saved,checkpoint)
            with patch.object(completion,'ROOT',project),\
                 patch('wheel_legged_gym_isaaclab.robot_model.validate_checkpoint_robot',return_value=runtime),\
                 patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'nonfinite'):completion.finish_joint_pd(state,True)
                self.assertFalse((state.parent/'final_delivery').exists())
                run.assert_not_called()

    def test_failed_extended_gate_never_calls_poweroff(self):
        with tempfile.TemporaryDirectory() as d:
            project,state,runtime,chain,report=self.fixture(Path(d))
            with patch.object(completion,'ROOT',project),patch.object(completion,'assess',return_value=report),\
                 patch.object(completion,'assess_stability',return_value={'passed':False}),\
                 patch('wheel_legged_gym_isaaclab.robot_model.validate_checkpoint_robot',return_value=runtime),\
                 patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'not stable'):completion.finish_joint_pd(state,True)
                run.assert_not_called()

    def test_overlap_or_changed_weights_or_changed_sources_cannot_poweroff(self):
        for change,message in (
            ('seeds','seeds overlap'),('weights','checkpoint differs'),('source','source snapshot changed')):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                project,state,runtime,chain,report=self.fixture(Path(d))
                if change=='seeds':
                    path=state.parent/'extended/manifest.json';m=json.loads(path.read_text())
                    m['seeds']=[83,284,285];path.write_text(json.dumps(m))
                elif change=='weights':report['checkpoint_sha256']='tampered'
                else:(project/'controller.py').write_text('# changed after validation')
                with patch.object(completion,'ROOT',project),patch.object(completion,'assess',return_value=report),\
                     patch.object(completion,'assess_stability',return_value={'passed':True}),\
                     patch.object(completion,'verify_asset_chain',return_value=(chain,{})),\
                     patch('wheel_legged_gym_isaaclab.robot_model.validate_checkpoint_robot',return_value=runtime),\
                     patch.object(completion.subprocess,'run') as run:
                    with self.assertRaisesRegex(RuntimeError,message):completion.finish_joint_pd(state,True)
                    run.assert_not_called()

class TestCompletionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Font discovery uses subprocess on first import. Initialize it before
        # mocking subprocess.run for poweroff tests, otherwise that mock also
        # intercepts matplotlib's binary fc-list response.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot

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
        runtime.update(leg_control_mode='implicit_joint_position',leg_joint_stiffness=300,leg_joint_damping=3,
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
        # Isolate the report builder as well as completion. Previously this
        # fixture regenerated the real project's "latest" learning documents.
        history_round=fixture_root/'mine_model/results'/root.name/'round_000'
        history_round.mkdir(parents=True)
        (history_round/'overrides.json').write_text('{}')
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
        with patch.object(completion,'ROOT',fixture_root),patch('build_training_record.ROOT', fixture_root),patch('build_training_record.MINE', fixture_root/'mine_model'),patch.object(sys,'argv',['complete','--state',str(state),'--shutdown']), patch.object(completion,'assess',return_value=report),patch.object(completion.subprocess,'run') as run:
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

    def test_changed_mcu_gravity_contract_blocks_delivery(self):
        for key, value in (('hardware_profile', 'series_balance_rtos_outer_20261010'),
                           ('feedforward_mode', 'mcu_pitch_compensated'),
                           ('feedforward_angle_bias', .03411),
                           ('feedforward_force_limit', 100.)):
            def mutate(root):
                path = root/'round/heldout_replay/resolved_env.json'
                config = json.loads(path.read_text())
                config[key] = value
                path.write_text(json.dumps(config))
            with self.subTest(field=key), tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, 'Controller differs'):
                    self.finalize_fixture(Path(d), mutation=mutate)

    def test_changed_external_effort_solver_contract_blocks_delivery(self):
        def mutate(root):
            path = root/'round/heldout_replay/resolved_env.json'
            config = json.loads(path.read_text())
            config['enable_external_forces_every_iteration'] = True
            path.write_text(json.dumps(config))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'Controller differs'):
                self.finalize_fixture(Path(d), mutation=mutate)

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

    def test_required_vmc_chain_rejects_old_implicit_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'status.json'
            state.write_text(json.dumps({'state':'accepted','overrides':{'leg_control_mode':'implicit_joint_position'}}))
            argv=['complete','--state',str(state),'--shutdown','--require-leg-control-mode','explicit_vmc']
            with patch.object(sys,'argv',argv),patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'wrong leg execution chain'):
                    completion.main()
                run.assert_not_called()

    def test_looser_absolute_tolerance_blocks_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            state,report=self.full_replay_fixture(root,[[43,44,45],[83,84,85],[83,84,85]])
            (root/'final_delivery_requirements.json').write_text(json.dumps({
                'relative_error_target':.2,'acceptance_criteria':{'speed_absolute_m_s':.01}}))
            argv=['complete','--state',str(state),'--shutdown']
            with patch.object(sys,'argv',argv),patch.object(completion,'assess',return_value=report),patch.object(completion.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'criterion was relaxed'):
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
