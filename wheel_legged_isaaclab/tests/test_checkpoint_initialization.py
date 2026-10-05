import unittest
import torch
from wheel_legged_gym_isaaclab.checkpoint_initialization import mirror_speed_command_initialization,center_leg_angle_initialization,scale_actor_feedback_initialization,scale_wheel_difference_initialization


class TestCommandWarmStart(unittest.TestCase):
    def test_network_parity_and_optimizer_coordinate_change(self):
        torch.manual_seed(43)
        state={'actor.0.weight':torch.randn(8,27),'critic.0.weight':torch.randn(8,27),
               'std':torch.full((6,),.1)}
        opt={'param_groups':[{'params':[0,1,2]}],
             'state':{i:{'exp_avg':torch.randn_like(v),'exp_avg_sq':torch.rand_like(v)}
                      for i,v in enumerate(state.values())}}
        source={'model_state_dict':state,'optimizer_state_dict':opt,'iter':2199}
        new=mirror_speed_command_initialization(source)
        obs=torch.randn(20,27);reflected=obs.clone();reflected[:,6].neg_()
        for i,key in enumerate(('actor.0.weight','critic.0.weight')):
            torch.testing.assert_close(obs@new['model_state_dict'][key].T,reflected@state[key].T)
            torch.testing.assert_close(new['optimizer_state_dict']['state'][i]['exp_avg_sq'],opt['state'][i]['exp_avg_sq'])
            torch.testing.assert_close(new['optimizer_state_dict']['state'][i]['exp_avg'][:,:6],opt['state'][i]['exp_avg'][:,:6])
            torch.testing.assert_close(new['optimizer_state_dict']['state'][i]['exp_avg'][:,6],-opt['state'][i]['exp_avg'][:,6])
        torch.testing.assert_close(new['model_state_dict']['std'],state['std'])
        self.assertEqual(new['iter'],2199)
        twice=mirror_speed_command_initialization(new)
        torch.testing.assert_close(twice['model_state_dict']['actor.0.weight'],state['actor.0.weight'])

    def test_normalized_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Normalized'):
            mirror_speed_command_initialization({'model_state_dict':{'actor_obs_normalizer.mean':torch.zeros(27)}})

    def test_leg_means_center_without_changing_other_actions_or_exploration(self):
        state={'actor.6.weight':torch.randn(6,8),'actor.6.bias':torch.randn(6),'std':torch.ones(6)*.1}
        source={'model_state_dict':state,'optimizer_state_dict':{
            'param_groups':[{'params':[0,1,2]}],
            'state':{i:{'exp_avg':torch.ones_like(v),'exp_avg_sq':torch.ones_like(v)} for i,v in enumerate(state.values())}}}
        new=center_leg_angle_initialization(source,.25,.26)
        x=torch.randn(10,8);before=x@state['actor.6.weight'].T+state['actor.6.bias']
        after=x@new['model_state_dict']['actor.6.weight'].T+new['model_state_dict']['actor.6.bias']
        torch.testing.assert_close(after[:,[0,3]],.25*before[:,[0,3]]+.75*.26)
        torch.testing.assert_close(after[:,[1,2,4,5]],before[:,[1,2,4,5]])
        torch.testing.assert_close(new['model_state_dict']['std'],state['std'])
        self.assertEqual(new['optimizer_state_dict']['state'][0]['exp_avg'][[0,3]].abs().sum().item(),0.)
        torch.testing.assert_close(new['optimizer_state_dict']['state'][0]['exp_avg'][[1,2,4,5]],torch.ones(4,8))

    def test_feedback_gain_trial_preserves_critic_exploration_and_other_optimizer_columns(self):
        torch.manual_seed(43)
        state={'actor.0.weight':torch.randn(8,27),'critic.0.weight':torch.randn(8,27),'std':torch.ones(6)*.1}
        source={'model_state_dict':state,'optimizer_state_dict':{
            'param_groups':[{'params':[0,1,2]}],
            'state':{i:{'exp_avg':torch.ones_like(v),'exp_avg_sq':torch.ones_like(v)} for i,v in enumerate(state.values())}}}
        new=scale_actor_feedback_initialization(source,[1],.75)
        obs=torch.randn(10,27);scaled=obs.clone();scaled[:,1]*=.75
        torch.testing.assert_close(obs@new['model_state_dict']['actor.0.weight'].T,scaled@state['actor.0.weight'].T)
        for key in ('critic.0.weight','std'):
            torch.testing.assert_close(new['model_state_dict'][key],state[key])
        other=[i for i in range(27) if i!=1]
        for name in ('exp_avg','exp_avg_sq'):
            self.assertEqual(new['optimizer_state_dict']['state'][0][name][:,1].abs().sum().item(),0.)
            torch.testing.assert_close(new['optimizer_state_dict']['state'][0][name][:,other],source['optimizer_state_dict']['state'][0][name][:,other])
        torch.testing.assert_close(source['optimizer_state_dict']['state'][0]['exp_avg'],torch.ones(8,27))

    def test_feedback_invalid_columns_rejected(self):
        for columns,scale in (([],.5),([27],.5),([1,1],.5),([1],0.)):
            with self.assertRaises(ValueError):scale_actor_feedback_initialization({},columns,scale)

    def test_wheel_difference_trial_keeps_mean_leg_actions_and_exploration(self):
        state={'actor.6.weight':torch.randn(6,8),'actor.6.bias':torch.randn(6),'std':torch.ones(6)*.1}
        source={'model_state_dict':state}
        new=scale_wheel_difference_initialization(source,.7)
        x=torch.randn(10,8)
        before=x@state['actor.6.weight'].T+state['actor.6.bias']
        after=x@new['model_state_dict']['actor.6.weight'].T+new['model_state_dict']['actor.6.bias']
        torch.testing.assert_close(after[:,2]+after[:,5],before[:,2]+before[:,5])
        torch.testing.assert_close(after[:,2]-after[:,5],.7*(before[:,2]-before[:,5]))
        torch.testing.assert_close(after[:,[0,1,3,4]],before[:,[0,1,3,4]])
        torch.testing.assert_close(new['model_state_dict']['std'],state['std'])


if __name__=='__main__':unittest.main()
