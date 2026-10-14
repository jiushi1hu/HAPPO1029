"""Latent-action likelihoods, cascade rewards, and hard annual storage closure."""
import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    _, config = load_configuration()
    data = SimpleNamespace(vmin=np.zeros(5), vmax=np.full(5, 86400000.),
        release_cap=np.full(5, 1000.), turbine_cap=np.full(5, 2.),
        power_cap=np.full(5, 10000.), head=np.full(5, 100.))
    return data, config, np.full(5, 43200000.)


class ActionRewardTests(unittest.TestCase):
    def setUp(self):
        self.data, self.config, self.storage = fixture()
        self.physics = ReservoirPhysics(self.data, self.config)

    def uniform_release(self, flow):
        release = np.full(5, flow)
        incoming = np.concatenate(([0.], release[:-1]))
        high = np.minimum(self.data.release_cap, incoming+self.storage/86400.)
        return self.physics.transition(self.storage, np.zeros(5), np.sqrt(release/high))

    def test_active_actions_are_invertible_and_not_clipped(self):
        outputs = []
        for value in (0., .1, .5, .9, 1.):
            row = self.physics.transition(self.storage, np.zeros(5), np.full(5, value))
            np.testing.assert_array_equal(row['actions'], np.full(5, value))
            np.testing.assert_array_equal(row['controls'], np.ones(5))
            inverse = np.sqrt((row['release']-row['low'])/(row['high']-row['low']))
            np.testing.assert_allclose(inverse, value, atol=1e-12)
            outputs.append(row['release'][0])
        self.assertTrue(np.all(np.diff(outputs)>0.))
        for bad in (-.01, 1.01, np.nan):
            with self.assertRaises(ValueError):
                self.physics.transition(self.storage, np.zeros(5), np.full(5, bad))

    def test_latent_likelihood_ratio_matches_release_density_ratio(self):
        u = torch.tensor([[.2],[.8]], dtype=torch.float64)
        old = torch.distributions.Beta(torch.tensor(2.),torch.tensor(3.))
        new = torch.distributions.Beta(torch.tensor(3.),torch.tensor(2.))
        width = torch.tensor([[100.],[300.]], dtype=torch.float64)
        jacobian = torch.log(2.*u*width)
        latent_ratio = (new.log_prob(u)-old.log_prob(u)).exp()
        release_ratio = ((new.log_prob(u)-jacobian)-(old.log_prob(u)-jacobian)).exp()
        torch.testing.assert_close(latent_ratio, release_ratio)

    def test_energy_reward_ordering_and_spill_saturation(self):
        zero, one, two, three = [self.uniform_release(q) for q in (0.,1.,2.,3.)]
        self.assertEqual(zero['reward'],0.)
        self.assertAlmostEqual(one['reward'],5.*21.1896/(24.*50000.),places=12)
        self.assertAlmostEqual(two['reward'],2.*one['reward'],places=12)
        self.assertAlmostEqual(three['reward'],two['reward'],places=12)
        self.assertGreater(three['spill'].sum(),two['spill'].sum())
        for row in (zero,one,two,three):
            self.assertTrue(0.<=row['reward']<=1.)
            self.assertAlmostEqual(row['reward']*self.physics.reference_energy,row['energy'].sum())

    def test_upstream_spill_can_supply_downstream_generation(self):
        self.data.turbine_cap[0] = 1.
        self.data.turbine_cap[1:] = 3.
        self.physics = ReservoirPhysics(self.data,self.config)
        one, two = self.uniform_release(1.),self.uniform_release(2.)
        self.assertAlmostEqual(two['energy'][0],one['energy'][0])
        self.assertGreater(two['spill'][0],one['spill'][0])
        self.assertGreater(two['reward'],one['reward'])


class TerminalStorageTests(unittest.TestCase):
    def env(self, mode='initial', external=None):
        data,config,storage=fixture()
        data.start_year=1965
        data.dates=(date(1965,1,1),date(1965,1,2))
        data.external=np.array([[1.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]) if external is None else external
        data.flow_mean=np.zeros(5);data.flow_std=np.ones(5)
        data.indices=lambda start,end:(0,2)
        data.initial_storage=lambda year:storage.copy()
        data.levels=lambda value:value/86400.
        config['terminal_storage_mode']=mode
        return CascadeReservoirEnv(data,config,start_year=1965,end_year=1965)

    def test_terminal_reachability_limits_current_actions_and_forces_final_closure(self):
        env=self.env();target=env.storage.copy()
        env.step(np.zeros(5))
        first=env.last_transition
        self.assertAlmostEqual(first['high'][0],1.,places=7)
        self.assertAlmostEqual(first['storage_end'][0]-target[0],86400.,places=5)
        _,_,rewards,done,_,_=env.step(np.ones(5))
        np.testing.assert_allclose(env.storage,target,atol=1e-4,rtol=0)
        np.testing.assert_allclose(env.last_transition['release'],np.ones(5),atol=1e-8)
        np.testing.assert_array_equal(env.last_transition['controls'],np.zeros(5))
        self.assertTrue(done.all())
        # Terminal closure is a constraint, not an extra reward or per-agent penalty.
        np.testing.assert_allclose(rewards,env.last_transition['energy'].sum()/env.physics.reference_energy)

    def test_zero_inflow_cannot_mine_initial_storage_with_terminal_closure(self):
        closed=self.env(external=np.zeros((2,5)));free=self.env(mode='free',external=np.zeros((2,5)))
        initial=closed.storage.copy()
        for _ in range(2):
            closed.step(np.ones(5));free.step(np.ones(5))
        np.testing.assert_allclose(closed.storage,initial,atol=1e-4,rtol=0)
        self.assertEqual(closed.last_transition['reward'],0.)
        self.assertLess(free.storage.sum(),initial.sum())

    def test_unreachable_target_raises_without_mutating_environment(self):
        env=self.env(external=np.zeros((2,5)))
        env.storage[0]-=86400.;before=env.storage.copy();index=env.index
        with self.assertRaisesRegex(InfeasibleCascade,'Terminal storage target is unreachable'):
            env.step(np.ones(5))
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,index);self.assertIsNone(env.last_transition)

    def test_annual_target_cannot_be_silently_applied_to_multiple_years(self):
        env=self.env()
        _,config=load_configuration()
        with self.assertRaisesRegex(ValueError,'single-year episode'):
            CascadeReservoirEnv(env.data,config,start_year=1965,end_year=1966)

    def test_solver_failure_is_not_misreported_as_physical_infeasibility(self):
        env=self.env();before=env.storage.copy()
        failure=SimpleNamespace(success=False,status=4,message='numerical solver failure')
        with patch('harl.envs.cascade_reservoir.terminal_storage.linprog',return_value=failure):
            with self.assertRaisesRegex(RuntimeError,'Terminal feasibility solver failed'):
                env.step(np.zeros(5))
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,env.start)


class RolloutActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.physical=load_configuration()
        cls.config['model']['hidden_sizes']=[16,16]
        cls.config['algo']['ppo_epoch']=cls.config['algo']['critic_epoch']=1
        cls.data=CascadeData(ROOT/'data/cascade_reservoir_new',cls.physical,cls.config['experiment'])

    def test_incompatible_action_and_forced_mask_configurations_are_rejected(self):
        for group,key,value in (('model','action_distribution','gaussian'),
            ('algo','use_policy_active_masks',False),('algo','mask_inactive_factors',False)):
            cfg=copy.deepcopy(self.config);cfg[group][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                CascadeReservoirRunner(self.data,self.physical,cfg,1)

    def test_actual_rollout_preserves_active_log_probabilities_and_masks_forced_factors(self):
        # Exercise both controllable and forced samples without spending an annual LP run here.
        data=copy.deepcopy(self.data);data.initial_storage_by_year[1965]=data.vmin.copy();data.external[0]=0.
        physical={**self.physical,'terminal_storage_mode':'free'}
        runner=CascadeReservoirRunner(data,physical,copy.deepcopy(self.config),1)
        runner.collect_episode(1965)
        for actor,buffer in zip(runner.actor,runner.actor_buffer):
            lp,_,_=actor.evaluate_actions(buffer.obs[:-1].reshape(-1,18),buffer.rnn_states[0],
                buffer.actions.reshape(-1,1),buffer.masks[:-1].reshape(-1,1))
            active=buffer.active_masks[:-1].reshape(-1)>0
            self.assertTrue(active.any());self.assertFalse(active.all())
            np.testing.assert_allclose(lp.detach().numpy()[active],buffer.action_log_probs.reshape(-1,1)[active],atol=1e-6)
            np.testing.assert_array_equal(buffer.action_log_probs.reshape(-1,1)[~active],0.)
            np.testing.assert_array_equal(buffer.actions.reshape(-1,1)[~active],.5)
        runner.prep_training();runner.train()
        for buffer in runner.actor_buffer:
            self.assertEqual(float(buffer.factor[0,0,0]),1.)


if __name__=='__main__':
    unittest.main()
