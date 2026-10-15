"""Change-of-variables and physical-mean checks for the release-fraction head."""
import unittest
import copy
from pathlib import Path
import numpy as np
import torch
from gym.spaces import Box
from examples.train_cascade import load_configuration
from harl.models.base.act import ACTLayer
from harl.models.base.distributions import BetaHead, FixedSqrtBeta


class ReleaseFractionTests(unittest.TestCase):
    def test_sampling_has_no_clipping_and_density_has_exact_jacobian(self):
        a,b=torch.tensor([[1.7],[3.]]),torch.tensor([[5.1],[2.]])
        base=torch.distributions.Beta(a,b);dist=FixedSqrtBeta(a,b)
        torch.manual_seed(19);x=base.sample((1024,))
        torch.manual_seed(19);u=dist.sample((1024,))
        torch.testing.assert_close(u,x.sqrt(),rtol=0,atol=0)
        torch.testing.assert_close(dist.log_probs(u),base.log_prob(u.square())+torch.log(2*u))
        self.assertTrue(((u>0)&(u<1)).all())
        for invalid in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                dist.log_prob(torch.full((2,1),invalid))

    def test_deterministic_release_is_the_expected_mapped_fraction(self):
        a,b=torch.tensor([[1.7],[3.],[100.]]),torch.tensor([[5.1],[2.],[25.]])
        dist=FixedSqrtBeta(a,b)
        expected=a/(a+b)
        torch.testing.assert_close(dist.mode().square(),expected)
        torch.testing.assert_close(dist.mean.square()+dist.variance,expected)
        # Different feasible intervals must retain the same expectation identity.
        lo=torch.tensor([[0.],[100.],[500.]])
        width=torch.tensor([[1000.],[7000.],[20.]])
        torch.testing.assert_close(lo+width*dist.mode().square(),lo+width*expected)

    def test_entropy_and_kl_match_transformed_samples(self):
        old=FixedSqrtBeta(torch.tensor([[2.]]),torch.tensor([[5.]]))
        new=FixedSqrtBeta(torch.tensor([[3.]]),torch.tensor([[4.]]))
        torch.manual_seed(27);u=old.sample((100000,))
        self.assertAlmostEqual(float(old.entropy().mean()),float(-old.log_probs(u).mean()),delta=.01)
        kl=torch.distributions.kl_divergence(old,new)
        self.assertAlmostEqual(float(kl.mean()),float((old.log_probs(u)-new.log_probs(u)).mean()),delta=.01)
        base_old=torch.distributions.Beta(old.concentration1,old.concentration0)
        base_new=torch.distributions.Beta(new.concentration1,new.concentration0)
        torch.testing.assert_close((new.log_probs(u)-old.log_probs(u)).exp(),
            (base_new.log_prob(u.square())-base_old.log_prob(u.square())).exp())

    def test_actor_sampling_evaluation_ratios_and_gradients_are_consistent(self):
        cfg,_=load_configuration()
        args={**cfg['model'],**cfg['algo'],'beta_action_coordinate':'release_fraction'}
        layer=ACTLayer(Box(0.,1.,(1,),dtype=np.float32),4,'orthogonal_',.01,args)
        obs=torch.zeros(200,4)
        u,old=layer(obs);new,entropy,dist=layer.evaluate_actions(obs,u)
        self.assertIsInstance(dist,FixedSqrtBeta)
        torch.testing.assert_close((new-old).exp(),torch.ones_like(old))
        torch.testing.assert_close(layer(obs,deterministic=True)[0],torch.full((200,1),.5))
        (-new.mean()-.001*entropy).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))
        with self.assertRaises(ValueError):
            BetaHead(4,1,'orthogonal_',.01,action_coordinate='invalid')

    def test_full_happo_update_and_diagnostics_preserve_transformed_policy(self):
        from harl.envs.cascade_reservoir.data_loader import CascadeData
        from harl.envs.cascade_reservoir.diagnostics import before_update, after_update
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        cfg, physical = load_configuration()
        cfg['model'].update(hidden_sizes=[16,16],beta_action_coordinate='release_fraction')
        cfg['algo'].update(ppo_epoch=1,critic_epoch=1)
        data = CascadeData(Path(__file__).resolve().parents[1]/'data/cascade_reservoir_new',physical,cfg['experiment'])
        outcomes = []
        for diagnostics in (False, True):
            runner = CascadeReservoirRunner(data,physical,copy.deepcopy(cfg),11)
            runner.collect_episode(1965)
            for actor, buffer in zip(runner.actor, runner.actor_buffer):
                lp,_,_=actor.evaluate_actions(buffer.obs[:-1].reshape(-1,18),buffer.rnn_states[0],
                    buffer.actions.reshape(-1,1),buffer.masks[:-1].reshape(-1,1))
                active=buffer.active_masks[:-1].reshape(-1)>0
                # Batched and single-observation float32 MLP kernels can differ
                # by a few ulps. Check the actual importance ratio against one.
                ratio=np.exp(lp.detach().numpy()[active]-buffer.action_log_probs.reshape(-1,1)[active])
                np.testing.assert_allclose(ratio,1.,atol=3e-6,rtol=0)
            before=before_update(runner) if diagnostics else None
            runner.prep_training(); runner.train()
            if diagnostics:
                report=after_update(runner,before)
                for actor in report['actors']:
                    self.assertAlmostEqual(actor['eval_mapping_fraction'],actor['expected_sampled_mapping_fraction'],places=6)
                    self.assertGreater(actor['parameter_delta_l2'],0.)
            outcomes.append(([copy.deepcopy(a.actor.state_dict()) for a in runner.actor]+
                [copy.deepcopy(runner.critic.critic.state_dict()),copy.deepcopy(runner.value_normalizer.state_dict())],
                torch.get_rng_state().clone()))
        for left,right in zip(outcomes[0][0],outcomes[1][0]):
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        torch.testing.assert_close(outcomes[0][1],outcomes[1][1],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
