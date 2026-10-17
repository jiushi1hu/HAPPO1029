"""Diagnostics must not change learning, random states, or physical accounting."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.diagnostics import (before_update, after_update, validation_report,
    fit_metrics, parameter_snapshot, parameter_update, monte_carlo_returns)
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT=Path(__file__).resolve().parents[1]


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.physical=load_configuration()
        cls.cfg['model']['hidden_sizes']=[16,16]
        for key in ['ppo_epoch','critic_epoch','actor_num_mini_batch','critic_num_mini_batch']:
            cls.cfg['algo'][key]=1
        cls.data=CascadeData(ROOT/'data/cascade_reservoir_new',cls.physical,cls.cfg['experiment'])

    def runner(self):
        return CascadeReservoirRunner(self.data,self.physical,copy.deepcopy(self.cfg),1)

    def test_diagnostics_preserve_actions_parameters_normalizer_and_rng(self):
        outcomes=[]
        for enabled in [False,True]:
            r=self.runner();r.collect_episode(1965)
            actions=[b.actions.copy() for b in r.actor_buffer]
            captured=before_update(r) if enabled else None
            r.prep_training();r.train()
            if enabled:
                report=after_update(r,captured)
                self.assertEqual(len(report['actors']),5)
                for i,a in enumerate(report['actors']):
                    self.assertTrue(0. <= a['expected_sampled_mapping_fraction'] <= 1.)
                    self.assertTrue(0. <= a['eval_mapping_fraction'] <= 1.)
                    b=r.actor_buffer[i]
                    with torch.no_grad():
                        _,_,dist=r.actor[i].evaluate_actions(b.obs[:-1,0],b.actions[:,0])
                        active=b.active_masks[:-1,0,0]>0
                        probs=dist.probs[active].numpy()
                        grid=np.arange(200)/199.
                        self.assertAlmostEqual(a['expected_sampled_mapping_fraction'],
                            float((probs*grid**2).sum(-1).mean()),places=6)
                        self.assertAlmostEqual(a['mean_u'],float((probs*grid).sum(-1).mean()),places=6)
                        self.assertAlmostEqual(a['eval_mapping_fraction'],
                            float(np.mean((probs.argmax(-1)/199.)**2)),places=6)
                    self.assertGreater(a['parameter_delta_l2'],0.)
                    self.assertEqual(a['unchanged_parameter_tensors'],[])
                    self.assertEqual(a['missing_gradient_tensors'],[])
                    self.assertGreater(a['final_grad_norm_post_clip'],0.)
                    self.assertLessEqual(a['final_grad_norm_post_clip'],.50001)
                    self.assertTrue(0.<=a['sampled_boundary_fraction']<=1.)
                self.assertGreater(report['critic_update']['parameter_delta_l2'],0.)
                self.assertEqual(report['gamma'],r.critic_buffer.gamma)
                self.assertAlmostEqual(report['episode_reward'],
                    float(r.critic_buffer.rewards.astype(np.float64).sum()))
                self.assertAlmostEqual(report['discounted_episode_reward'],float(captured['mc'][0,0,0]))
                self.assertTrue(np.isfinite(report['critic_mc_post']['rmse']))
            outcomes.append(dict(actions=actions,actors=[copy.deepcopy(a.actor.state_dict()) for a in r.actor],
                critic=copy.deepcopy(r.critic.critic.state_dict()),norm=copy.deepcopy(r.value_normalizer.state_dict()),
                rng=torch.get_rng_state().clone()))
        a,b=outcomes
        for x,y in zip(a['actions'],b['actions']):np.testing.assert_array_equal(x,y)
        for x,y in zip(a['actors']+[a['critic'],a['norm']],b['actors']+[b['critic'],b['norm']]):
            for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        torch.testing.assert_close(a['rng'],b['rng'],rtol=0,atol=0)

    def test_validation_sampling_preserves_training_rng(self):
        r=self.runner();r.algo_args['experiment']['diagnostic_stochastic_repeats']=2
        state=torch.get_rng_state().clone()
        def fake(*args,**kwargs):
            return dict(energy_TWh=float(torch.rand(())))
        rows=[dict(energy_TWh=1.) for _ in self.data.validation_years]
        with patch('harl.envs.cascade_reservoir.diagnostics.evaluate_policy',side_effect=fake) as evaluate,redirect_stdout(io.StringIO()):
            a=validation_report(r,rows)
            b=validation_report(r,rows)
            self.assertEqual(evaluate.call_count,20)
        self.assertEqual(a,b)
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)

    def test_validation_accounting_and_repeatability(self):
        r=self.runner()
        rows=[]
        for enabled in [False,True]:
            rows.append(evaluate_policy(r.actor,self.data,self.physical,2024,1,0,0,diagnostics=enabled))
        for key,value in rows[0].items():self.assertEqual(value,rows[1][key])
        d=rows[1]
        self.assertAlmostEqual(sum(d['energy_by_reservoir_TWh']),d['energy_TWh'],places=9)
        self.assertAlmostEqual(sum(d['spill_by_reservoir_billion_m3']),d['spill_volume_m3']/1e9,places=9)
        self.assertTrue(all(-1e-7<=v<=1+1e-7 for v in d['end_storage_fraction']))

    def test_mc_fit_metrics_are_not_gae_metrics(self):
        good=fit_metrics([1.,2.,3.],[1.,2.,3.])
        self.assertEqual(good['rmse'],0.)
        self.assertEqual(good['explained_variance'],1.)
        constant=fit_metrics([1.,2.,3.],[2.,2.,2.])
        self.assertEqual(constant['explained_variance'],0.)

    def test_mc_returns_use_selected_gamma_and_stop_at_terminal_boundaries(self):
        rewards=np.array([1.,2.,3.,4.])[:,None,None]
        masks=np.array([1.,1.,0.,1.,0.])[:,None,None]
        for gamma,expected in ((0.,[1.,2.,3.,4.]),(.5,[2.,2.,5.,4.]),(1.,[3.,2.,7.,4.])):
            with self.subTest(gamma=gamma):
                np.testing.assert_array_equal(monte_carlo_returns(rewards,masks,gamma)[:,0,0],expected)

    def test_real_rollout_mc_diagnostics_match_discounted_rewards(self):
        r=self.runner()
        r.collect_episode(1968)
        before=before_update(r)
        cb=r.critic_buffer
        rewards=cb.rewards.astype(np.float64)
        expected=np.stack([(rewards[t:] *
            (cb.gamma ** np.arange(len(rewards)-t))[:,None,None]).sum(axis=0)
            for t in range(len(rewards))])
        np.testing.assert_allclose(before['mc'],expected,rtol=1e-12,atol=1e-12)

    def test_nonfinite_rollout_is_rejected_with_location(self):
        r=self.runner();r.collect_episode(1965)
        r.actor_buffer[2].action_log_probs[0,0,0]=np.nan
        with self.assertRaisesRegex(FloatingPointError,'actor 2 buffer action_log_probs'):
            before_update(r)

    def test_parameter_diagnostic_detects_stalled_and_nonfinite_gradients(self):
        module=torch.nn.Linear(2,1)
        snapshot=parameter_snapshot(module)
        stalled=parameter_update(module,snapshot)
        self.assertEqual(stalled['parameter_delta_l2'],0.)
        self.assertEqual(stalled['unchanged_parameter_tensors'],['weight','bias'])
        self.assertEqual(stalled['missing_gradient_tensors'],['weight','bias'])
        module.weight.grad=torch.zeros_like(module.weight)
        self.assertEqual(parameter_update(module,snapshot)['zero_gradient_tensors'],['weight'])
        module.weight.grad[0,0]=float('inf')
        with self.assertRaisesRegex(FloatingPointError,'gradient weight'):
            parameter_update(module,snapshot)


if __name__=='__main__':unittest.main()
