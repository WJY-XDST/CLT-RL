"""Fresh pilots and interrupted pilots must not mix checkpoint contracts."""
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import auto_train_mine as training


class TestCooperativeCheckpointStop(unittest.TestCase):
    def test_stop_saves_complete_runner_only_after_the_logging_boundary(self):
        from types import SimpleNamespace
        from wheel_legged_gym_isaaclab.training_state import install_checkpoint_stop, CheckpointStopRequested
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'STOP';events=[]
            runner=SimpleNamespace(current_learning_iteration=2156,
                log=lambda data:events.append(('logged',data)),save=lambda path:events.append(('saved',path)))
            install_checkpoint_stop(runner,marker,d)
            runner.log({'it':2155})
            self.assertEqual([e[0] for e in events],['logged'])
            marker.touch()
            with self.assertRaises(CheckpointStopRequested):runner.log({'it':2156})
            self.assertEqual([e[0] for e in events],['logged','logged','saved'])
            self.assertEqual(Path(events[-1][1]).name,'model_2156_interrupted.pt')

    def test_no_stop_file_does_not_wrap_the_runner(self):
        from types import SimpleNamespace
        from wheel_legged_gym_isaaclab.training_state import install_checkpoint_stop
        runner=SimpleNamespace(log=Mock())
        original=runner.log;install_checkpoint_stop(runner,None,'unused')
        self.assertIs(runner.log,original)

    def test_exec_stop_cannot_write_to_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):training.request_checkpoint_stop(Path(d))
            self.assertFalse((Path(d)/'STOP').exists())


class TestPilotInitialization(unittest.TestCase):
    def launch_command(self, recovered, safe_standing=False, motion=False, fine_tuning=False, wheel_precision=False, num_envs=None, full_policy=False, reset_full_optimizer=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);logs=root/'logs';logs.mkdir()
            output=root/'new_experiment'
            if safe_standing:
                import json
                output.mkdir()
                best=root/'best.json';best.write_text(json.dumps({
                    'acceptance_criteria': training.acceptance_criteria(),
                    'conditions':[{'passed':False,'survivors':3,'trials':[{}]*3}]}))
                incumbent=root/'model_999.pt';incumbent.write_bytes(b'stable policy')
                (output/'status.json').write_text(json.dumps({'state':'planning_next_round','round':1,
                    'checkpoint':str(incumbent),'best_checkpoint':str(incumbent),'best_assessment':str(best),
                    **({'agent_overrides':{'algorithm.entropy_coef':.001}} if fine_tuning else {}),
                    **({'wheel_head_only_finetune':True} if wheel_precision else {}),
                    **({'full_policy_reset_optimizer_pending':True} if reset_full_optimizer else {}),
                    **({'overrides':{'leg_control_mode':'explicit_vmc','commands.standing_only_steps':48000}} if motion else {})}))
            checkpoint=None
            if recovered:
                run=logs/('20261004_mine_auto_r001_new_experiment' if safe_standing else '20261004_mine_auto_r000_new_experiment');run.mkdir()
                checkpoint=run/('model_999.pt' if safe_standing else 'model_5.pt');checkpoint.write_bytes(b'saved weights and optimizer')
            child=Mock(pid=123);child.wait.return_value=1
            argv=['train','--fresh','--trial_target','10','--output',str(output)]
            if num_envs is not None:argv+=['--num_envs',str(num_envs)]
            if full_policy:argv+=['--full-policy-training']
            with patch.object(training,'ROOT',root),patch.object(training,'LOG_ROOT',logs),patch.object(sys,'argv',argv),patch.object(training.subprocess,'Popen',return_value=child) as start:
                with self.assertRaisesRegex(RuntimeError,'Job failed'):
                    training.main()
                command=start.call_args.args[0]
                return command,str(checkpoint)

    def test_fresh_pilot_does_not_load_an_existing_policy(self):
        command,_=self.launch_command(False)
        self.assertNotIn('--checkpoint_path',command)

    def test_fresh_vmc_pilot_pins_actual_execution_mode(self):
        command,_=self.launch_command(False)
        self.assertEqual(command[command.index('--require_leg_control_mode')+1],'explicit_vmc')
        self.assertIn('env.leg_control_mode="explicit_vmc"',command)

    def test_supported_vmc_resume_remains_explicitly_identified(self):
        command,_=self.launch_command(False,safe_standing=True,motion=True)
        self.assertEqual(command[command.index('--require_leg_control_mode')+1],'explicit_vmc')

    def test_execution_mode_switch_cannot_reuse_existing_experiment(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);logs=root/'logs';logs.mkdir();output=root/'old';output.mkdir()
            (output/'status.json').write_text(json.dumps({'overrides':{'leg_control_mode':'implicit_joint_position'}}))
            argv=['train','--fresh','--output',str(output),'--leg-control-mode','explicit_vmc']
            with patch.object(training,'ROOT',root),patch.object(training,'LOG_ROOT',logs),patch.object(sys,'argv',argv),patch.object(training.subprocess,'Popen') as start:
                with self.assertRaisesRegex(ValueError,'new --output'):
                    training.main()
                start.assert_not_called()

    def test_large_environment_count_reaches_resumed_training_command(self):
        command,_=self.launch_command(False,safe_standing=True,wheel_precision=True,num_envs=12288)
        self.assertEqual(command[command.index('--num_envs')+1],'12288')
        self.assertIn('--checkpoint_path',command)
        self.assertNotIn('--reset_optimizer',command)

    def test_interrupted_pilot_restores_weights_and_optimizer(self):
        command,checkpoint=self.launch_command(True)
        self.assertEqual(command[command.index('--checkpoint_path')+1],checkpoint)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_safe_standing_with_tracking_error_keeps_optimizer_and_exploration(self):
        command,_=self.launch_command(False,safe_standing=True)
        self.assertIn('--checkpoint_path',command)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_vmc_motion_transition_does_not_apply_retired_ik_exploration_override(self):
        command,_=self.launch_command(False,safe_standing=True,motion=True)
        self.assertNotIn('--wheel_exploration_std',command)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_precision_phase_changes_entropy_without_resetting_weights_or_std(self):
        command,_=self.launch_command(False,safe_standing=True,fine_tuning=True)
        self.assertIn('agent.algorithm.entropy_coef=0.001',command)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_retired_wheel_precision_state_cannot_freeze_restarted_policy(self):
        command,_=self.launch_command(False,safe_standing=True,fine_tuning=True,wheel_precision=True)
        self.assertNotIn('--wheel_head_only_finetune',command)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_retired_precision_state_keeps_full_vmc_motion_training(self):
        command,_=self.launch_command(False,safe_standing=True,motion=True,wheel_precision=True)
        self.assertNotIn('--wheel_head_only_finetune',command)
        self.assertNotIn('--wheel_exploration_std',command)
        self.assertNotIn('--reset_optimizer',command)

    def test_explicit_full_policy_overrides_persisted_wheel_only_mode(self):
        command,_=self.launch_command(False,safe_standing=True,fine_tuning=True,
                                      wheel_precision=True,full_policy=True)
        self.assertNotIn('--wheel_head_only_finetune',command)
        self.assertIn('--checkpoint_path',command)
        self.assertIn('agent.algorithm.entropy_coef=0.001',command)
        self.assertNotIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)

    def test_full_policy_is_default_without_saved_precision_mode(self):
        command,_=self.launch_command(False,safe_standing=True)
        self.assertNotIn('--wheel_head_only_finetune',command)

    def test_explicit_full_policy_optimizer_transition_keeps_weights_and_noise(self):
        command,_=self.launch_command(False,safe_standing=True,fine_tuning=True,
                                      wheel_precision=True,full_policy=True,reset_full_optimizer=True)
        self.assertIn('--checkpoint_path',command)
        self.assertIn('--reset_optimizer',command)
        self.assertNotIn('--reset_exploration_std',command)
        self.assertNotIn('--wheel_head_only_finetune',command)

    def test_same_iteration_recovery_keeps_this_branch_and_its_new_optimizer(self):
        command,checkpoint=self.launch_command(True,safe_standing=True,full_policy=True,
                                              reset_full_optimizer=True)
        self.assertEqual(command[command.index('--checkpoint_path')+1],checkpoint)
        self.assertNotIn('--reset_optimizer',command)


class TestMeasuredPlanning(unittest.TestCase):
    def test_failed_validation_drives_next_round_instead_of_passing_short_replay(self):
        import json
        for failed_stage in ('heldout_replay', 'export_replay'):
            with self.subTest(stage=failed_stage), tempfile.TemporaryDirectory() as d:
                root=Path(d);logs=root/'logs';logs.mkdir()
                output=root/'experiment'
                run=logs/'date_mine_auto_r000_experiment';run.mkdir()
                (run/'model_9.pt').write_bytes(b'candidate')
                short={'passed':True,'criterion':'unchanged','survivors':1,'resets':0,'total_trials':1,
                       'conditions':[{'survivors':1,'trials':[{}]}]}
                failed={**short,'passed':False,'failure_stage':failed_stage}
                reports=[short,failed] if failed_stage=='heldout_replay' else [short,short,failed]
                child=Mock(pid=123);child.wait.return_value=0
                with patch.object(training,'ROOT',root), patch.object(training,'LOG_ROOT',logs), \
                     patch.object(sys,'argv',['train','--fresh','--trial_target','10','--max_rounds','1','--output',str(output)]), \
                     patch.object(training.subprocess,'Popen',return_value=child) as launch, \
                     patch.object(training,'assess',side_effect=reports), patch.object(training,'plot'), \
                     patch.object(training,'quality',return_value=(1,)), \
                     patch.object(training,'plan_after',return_value=({},'use measured failure')) as plan:
                    training.main()
                self.assertIs(plan.call_args.args[0],failed)
                state=json.loads((output/'status.json').read_text())
                self.assertFalse(state['passed'])
                self.assertIn(failed_stage,state['last_assessment'])
                self.assertEqual(launch.call_count,len(reports))

    def test_capped_tracking_with_measured_chatter_tunes_smoothness_without_relaxing_acceptance(self):
        keys=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg','leg_angle_mae_rad','leg_length_mae_m','wheel_speed_mae_rad_s')
        trial={'metrics':{k:0. for k in keys},'limits':{k:1. for k in keys},
               'diagnostics':{'wheel_reference_delta_rms_rad_s':2.}}
        trial['metrics']['wheel_speed_mae_rad_s']=.5;trial['limits']['wheel_speed_mae_rad_s']=.3
        report={'conditions':[{'passed':False,'survivors':3,'speed':0.,'yaw':0.,'trials':[trial]*3}]}
        overrides={'commands.standing_only_steps':67200,'rewards.standing_wheel_tracking':-10.,'rewards.action_rate':-.5}
        changed,purpose=training.plan_after(report,overrides,5000)
        self.assertEqual(changed['rewards.standing_wheel_tracking'],-10.)
        self.assertEqual(changed['rewards.action_rate'],-.75)
        self.assertEqual(trial['limits']['wheel_speed_mae_rad_s'],.3)
        self.assertIn('measured wheel reference',purpose)
        self.assertEqual(training.preserve_tracking_signal(changed)['rewards.action_rate_penalty_clip'],18.)
        trial['diagnostics']['wheel_reference_delta_rms_rad_s']=.5
        changed,_=training.plan_after(report,overrides,5000)
        self.assertEqual(changed['rewards.action_rate'],-.5)

    def test_warm_start_branch_cannot_recover_higher_old_run_checkpoint(self):
        state={'run_name_suffix':'_speed_angle_init','iteration_offset':-400}
        target,name=training.round_contract(state,1000,200,9,'experiment')
        self.assertEqual(target,2400)
        self.assertEqual(name,'mine_auto_r009_experiment_speed_angle_init')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old=root/'date_mine_auto_r009_experiment';old.mkdir()
            (old/'model_2699.pt').touch()
            with patch.object(training,'LOG_ROOT',root):
                self.assertEqual(training.checkpoints_for(name),[])
        self.assertEqual(training.round_contract(state,1000,200,10,'experiment')[0],2600)

    def test_tracking_tuning_retains_signal_for_reverse_running_and_turns(self):
        overrides={'rewards.lin_vel_error_sq':-37.5,'rewards.yaw_rate_error_sq':-7.8125,
                   'rewards.standing_wheel_tracking':-1.220703125,'leg_effort_limit':10.}
        changed=training.preserve_tracking_signal(overrides)
        # At +0.6 m/s, standing and running backwards must have different
        # costs. The old common -1 reward/s cap erased that distinction.
        def cost(error,weight,clip):return min(abs(weight)*error**2,clip)
        self.assertGreater(cost(1.2,-37.5,changed['rewards.lin_vel_penalty_clip']),
                           cost(.6,-37.5,changed['rewards.lin_vel_penalty_clip']))
        self.assertGreater(cost(.6,-7.8125,changed['rewards.yaw_rate_penalty_clip']),
                           cost(.3,-7.8125,changed['rewards.yaw_rate_penalty_clip']))
        self.assertEqual(changed['leg_effort_limit'],10.)
        self.assertNotIn('rewards.lin_vel_penalty_clip',overrides)
        self.assertLess(changed['rewards.termination'],-60.)
        self.assertGreater(changed['rewards.orientation_penalty_clip'],150*.25**2)

    def test_smoothness_tuning_can_distinguish_full_and_partial_wheel_chatter(self):
        changed=training.preserve_tracking_signal({'rewards.action_rate':-.5})
        clip=changed['rewards.action_rate_penalty_clip']
        # Both wheels toggling +/-1 have squared change 8. A half-sized
        # toggle has change 2; a shared cap of 1 made both cost exactly 1.
        self.assertGreater(min(.5*8,clip),min(.5*2,clip))
        self.assertEqual(clip,12.)

    def test_safe_standing_wheel_error_changes_tracking_not_death_penalty(self):
        metrics={key:0. for key in ('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg',
                                  'leg_angle_mae_rad','leg_length_mae_m','wheel_speed_mae_rad_s')}
        metrics['wheel_speed_mae_rad_s']=.46
        limits={key:1. for key in metrics};limits['wheel_speed_mae_rad_s']=.3
        trial={'metrics':metrics,'limits':limits}
        report={'conditions':[{'passed':False,'survivors':3,'trials':[trial]*3}]}
        overrides={'leg_control_mode':'explicit_vmc','wheel_damping':.5,'rewards.termination':-12}
        changed,purpose=training.plan_after(report,overrides,1000)
        self.assertEqual(changed['rewards.termination'],-12)
        self.assertEqual(changed['wheel_damping'],.5)
        self.assertEqual(changed['rewards.standing_wheel_tracking'],-.625)
        self.assertEqual(changed['commands.standing_only_steps'],96000)
        self.assertIn('standing_wheel_tracking',purpose)

    def test_best_policy_comparison_includes_wheel_tracking(self):
        def report(wheel_error,speed_error):
            passed=wheel_error<=.3
            trial={'first_failure_s':None,'passed':passed,'metrics':{'speed_mae_m_s':speed_error,'wheel_speed_mae_rad_s':wheel_error},
                   'limits':{'speed_mae_m_s':.05,'wheel_speed_mae_rad_s':.3}}
            return {'total_trials':3,'survivors':3,'conditions':[{'passed':passed,'survivors':3,'trials':[trial]*3}]}
        self.assertGreater(training.quality(report(.2,.001)),training.quality(report(.5,0.)))

    def test_near_tolerance_running_ranks_above_stationary_policy_with_large_motion_error(self):
        def report(ratios):
            trials=[{'first_failure_s':None,'passed':x<=1.,'metrics':{'speed_mae_m_s':x},'limits':{'speed_mae_m_s':1.}} for x in ratios]
            return {'total_trials':len(trials),'survivors':len(trials),'conditions':[
                {'passed':t['passed'],'survivors':1,'trials':[t]} for t in trials]}
        self.assertGreater(training.quality(report([1.1,1.2,1.3])),training.quality(report([.1,5.,9.])))

    def test_interrupt_checkpoint_is_recoverable_and_backup_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'date_run';run.mkdir()
            for name in ('model_100.pt','model_137_interrupted.pt','model_999_backup.pt'):(run/name).touch()
            with patch.object(training,'LOG_ROOT',root):
                found=training.checkpoints_for('run')
                newest=max(found,key=training.checkpoint_iteration)
                self.assertEqual(newest.name,'model_137_interrupted.pt')
                self.assertEqual(len(found),2)

    def test_small_stance_tracking_regression_does_not_erase_motion_course(self):
        keys=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg','leg_angle_mae_rad','leg_length_mae_m','wheel_speed_mae_rad_s')
        trial={'metrics':{k:0. for k in keys},'limits':{k:1. for k in keys}}
        trial['metrics']['wheel_speed_mae_rad_s']=.35;trial['limits']['wheel_speed_mae_rad_s']=.3
        report={'conditions':[{'passed':False,'survivors':3,'trials':[trial]*3}]}
        overrides={'leg_control_mode':'explicit_vmc','commands.standing_only_steps':67200}
        changed,_=training.plan_after(report,overrides,1600)
        self.assertEqual(changed['commands.standing_only_steps'],67200)
        self.assertIn('rewards.standing_wheel_tracking',changed)

    def test_safe_stance_does_not_hide_larger_motion_tracking_error(self):
        keys=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg','leg_angle_mae_rad','leg_length_mae_m','wheel_speed_mae_rad_s')
        stand={'metrics':{k:0. for k in keys},'limits':{k:1. for k in keys}}
        stand['metrics']['wheel_speed_mae_rad_s']=.9;stand['limits']['wheel_speed_mae_rad_s']=.3
        motion={'metrics':{k:0. for k in keys},'limits':{k:1. for k in keys}}
        motion['metrics']['speed_mae_m_s']=1.09;motion['limits']['speed_mae_m_s']=.12
        report={'conditions':[{'speed':0.,'yaw':0.,'passed':False,'survivors':3,'trials':[stand]*3},
                              {'speed':.6,'yaw':0.,'trials':[motion]*3}]}
        changed,_=training.plan_after(report,{'commands.standing_only_steps':67200},2200)
        self.assertEqual(changed['rewards.lin_vel_error_sq'],-37.5)
        self.assertNotIn('rewards.standing_wheel_tracking',changed)

    def test_safe_motion_learning_is_not_rolled_back_for_tracking_error(self):
        report={'survivors':27,'total_trials':27,'passed':False}
        best={'survivors':27,'total_trials':27,'passed':False}
        self.assertEqual(training.training_checkpoint('latest','old_stance',report,best),'latest')
        report['survivors']=26
        self.assertEqual(training.training_checkpoint('latest','old_stance',report,best),'old_stance')

    def test_unseen_motion_failure_keeps_curriculum_but_stance_failure_rolls_back(self):
        stance={'survivors':3,'trials':[{}]*3}
        report={'survivors':26,'total_trials':27,'conditions':[stance]}
        best={'survivors':27,'total_trials':27}
        self.assertEqual(training.training_checkpoint('latest','old',report,best,False),'latest')
        self.assertEqual(training.training_checkpoint('latest','old',report,best,True),'old')
        stance['survivors']=2
        self.assertEqual(training.training_checkpoint('latest','old',report,best,False),'old')
