"""Mean-preserving exploration control and HAPPO likelihood consistency."""
import unittest
import numpy as np
import torch
from gym.spaces import Box
from harl.models.base.act import ACTLayer
from harl.models.base.distributions import BetaHead, FixedNormal
from examples.train_cascade import load_configuration


class ConcentrationTests(unittest.TestCase):
    def test_mean_preserved_and_variance_bounded_with_same_weights(self):
        torch.manual_seed(3)
        original=BetaHead(4,1,'orthogonal_',.01)
        controlled=BetaHead(4,1,'orthogonal_',.01,98.)
        controlled.load_state_dict(original.state_dict())
        x=torch.randn(100,4)
        old,new=original(x),controlled(x)
        torch.testing.assert_close(old.mean,new.mean,rtol=2e-7,atol=1e-7)
        self.assertTrue((new.variance<old.variance).all())
        self.assertTrue((new.variance.sqrt()<.05).all())
        torch.testing.assert_close(new.concentration0+new.concentration1,
                                   old.concentration0+old.concentration1+98.,rtol=2e-7,atol=1e-5)
        self.assertTrue((new.concentration0>=1).all() and (new.concentration1>=1).all())

    def test_zero_offset_matches_original_distribution_exactly(self):
        head=BetaHead(4,1,'orthogonal_',.01,0.)
        x=torch.randn(50,4)
        alpha,beta=(torch.nn.functional.softplus(head.linear(x))+1.).chunk(2,-1)
        dist=head(x)
        torch.testing.assert_close(alpha,dist.concentration1,rtol=0,atol=0)
        torch.testing.assert_close(beta,dist.concentration0,rtol=0,atol=0)

    def test_controlled_density_is_used_for_sampling_and_ratio(self):
        cfg,_=load_configuration()
        args={**cfg['model'],**cfg['algo'],'beta_concentration_offset':98.}
        layer=ACTLayer(Box(0.,1.,(1,),dtype=np.float32),4,'orthogonal_',.01,args)
        x=torch.zeros(100,4)
        actions,old=layer(x)
        new,entropy,dist=layer.evaluate_actions(x,actions)
        torch.testing.assert_close((new-old).exp(),torch.ones_like(old),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(layer(x,deterministic=True)[0],dist.mean)
        (-new.mean()-.001*entropy).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None))
        self.assertTrue(((actions>0)&(actions<1)).all())

    def test_gaussian_default_unchanged_and_invalid_offsets_rejected(self):
        cfg,_=load_configuration();args={**cfg['model'],**cfg['algo']}
        args.pop('action_distribution')
        layer=ACTLayer(Box(-1.,1.,(1,),dtype=np.float32),4,'orthogonal_',.01,args)
        self.assertIsInstance(layer.action_out(torch.zeros(1,4)),FixedNormal)
        for bad in [-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):BetaHead(4,1,'orthogonal_',.01,bad)

    def test_lambda_one_uses_complete_annual_return_without_critic_bootstrap(self):
        from pathlib import Path
        from harl.envs.cascade_reservoir.data_loader import CascadeData
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        cfg,physical=load_configuration()
        cfg['algo']['gae_lambda']=1.
        cfg['model']['hidden_sizes']=[16,16]
        cfg['model']['beta_concentration_offset']=98.
        data=CascadeData(Path(__file__).resolve().parents[1]/'data/cascade_reservoir_new',physical,cfg['experiment'])
        r=CascadeReservoirRunner(data,physical,cfg,1)
        r.value_normalizer.update(np.array([[20.],[40.]],dtype=np.float32))
        r.collect_episode(1965)
        rewards=r.critic_buffer.rewards
        expected=np.cumsum(rewards[::-1].astype(np.float64),axis=0)[::-1]
        np.testing.assert_allclose(r.critic_buffer.returns[:-1],expected,rtol=1e-5,atol=1e-4)


if __name__=='__main__':unittest.main()
