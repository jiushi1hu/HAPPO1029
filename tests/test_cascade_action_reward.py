"""Discrete-action likelihoods, cascade rewards, and hard annual storage closure."""
import copy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    _, config = load_configuration()
    config['head_model'] = 'fixed'  # This toy system has no storage-elevation curve.
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

    def test_index_likelihood_ratio_matches_mapped_release_probability_ratio(self):
        grid = torch.arange(200, dtype=torch.float64) / 199.
        old = torch.distributions.Categorical(logits=grid)
        new = torch.distributions.Categorical(logits=-grid)
        indices = torch.tensor([0, 20, 100, 199])
        for low, width in ((0., 100.), (50., 300.)):
            releases = low + width * grid.square()
            # At fixed noncollapsed bounds each release corresponds to one index.
            recovered = torch.searchsorted(releases, releases[indices])
            release_ratio = new.probs[recovered] / old.probs[recovered]
            index_ratio = (new.log_prob(indices) - old.log_prob(indices)).exp()
            torch.testing.assert_close(index_ratio, release_ratio)

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


class DiscreteActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = load_configuration()
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new',cls.physical,cls.config['experiment'])

    def test_all_200_indices_quadratic_mapping_and_inverse(self):
        self.assertEqual(self.config['model']['action_distribution'],'categorical')
        self.assertEqual(self.config['model']['action_bins'],200)
        env = CascadeReservoirEnv(self.data,self.physical,start_year=1965)
        self.assertTrue(all(space.n == 200 for space in env.action_space))
        storage = .5*(self.data.vmin+self.data.vmax)
        releases = []
        for index in range(200):
            u = env.normalize_actions(np.full(5,index))
            np.testing.assert_array_equal(u,np.full(5,index/199.))
            row = env.physics.transition(storage,np.zeros(5),u)
            np.testing.assert_allclose(row['release'],row['low']+(row['high']-row['low'])*(index/199.)**2)
            np.testing.assert_array_equal(env.actions_from_releases(row['release'],row['low'],row['high']),
                                          np.full(5,index))
            releases.append(row['release'][0])
        self.assertTrue(np.all(np.diff(releases)>0))
        zero = np.zeros(5)
        np.testing.assert_array_equal(env.actions_from_releases(zero,zero,zero),zero)
        # Inverse quantization is in normalized control coordinates.
        np.testing.assert_array_equal(env.actions_from_releases(np.full(5,(20.2/199.)**2),zero,np.ones(5)),
                                      np.full(5,20))

    def test_invalid_indices_fail_without_state_mutation(self):
        for count in (None,0,1,2.5,True):
            with self.assertRaises(ValueError):
                CascadeReservoirEnv(self.data,self.physical,start_year=1965,action_bins=count)
        env = CascadeReservoirEnv(self.data,self.physical,start_year=1965,action_bins=200)
        before = env.storage.copy()
        for index in (-1,200,.5,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                env.step(np.full(5,index))
            np.testing.assert_array_equal(env.storage,before)
            self.assertEqual(env.index,env.start)

    def test_categorical_rollout_update_and_frozen_evaluation(self):
        from harl.envs.cascade_reservoir.diagnostics import before_update, after_update
        from harl.envs.cascade_reservoir.evaluation import evaluate_policy
        from harl.models.base.distributions import FixedCategorical
        r = CascadeReservoirRunner(self.data,self.physical,copy.deepcopy(self.config),1)
        for year in (1965,1968):
            r.collect_episode(year)
            self.assertEqual(r.length,365 if year==1965 else 366)
            for actor,b in zip(r.actor,r.actor_buffer):
                self.assertEqual(b.available_actions.shape,(r.length+1,1,200))
                self.assertTrue(np.all(b.actions==np.floor(b.actions)))
                self.assertTrue(np.all((b.actions>=0)&(b.actions<200)))
                lp,_,dist = actor.evaluate_actions(b.obs[:-1,0],b.actions[:,0],
                                                    b.available_actions[:-1,0])
                self.assertIsInstance(dist,FixedCategorical)
                self.assertEqual(dist.probs.shape,(r.length,200))
                # Density is for the chosen index, not for i/199 or the release.
                manual = dist.probs.gather(1,torch.from_numpy(b.actions[:,0]).long()).log()
                torch.testing.assert_close(lp,manual,atol=1e-6,rtol=1e-6)
                active = b.active_masks[:-1,0,0]>0
                np.testing.assert_allclose(np.exp(lp.detach().numpy()[active]-b.action_log_probs[:,0][active]),
                                           1.,atol=1e-5,rtol=0)
                np.testing.assert_array_equal(b.actions[:,0][~active],0)
                np.testing.assert_array_equal(b.action_log_probs[:,0][~active],0)
                actions = actor.act(b.obs[:1,0],deterministic=True)
                np.testing.assert_array_equal(actions.detach().numpy(),dist.probs[:1].argmax(-1,keepdim=True).numpy())
            before = before_update(r)
            r.prep_training();r.train()
            report = after_update(r,before)
            for item in report['actors']:
                self.assertGreater(item['parameter_delta_l2'],0)
                self.assertTrue(np.isfinite(item['policy_kl']))
                self.assertLess(item['ratio_max'],10.)
                self.assertTrue(0<=item['expected_sampled_mapping_fraction']<=1)
            np.testing.assert_allclose(r.envs[0].storage,self.data.initial_storage(year),atol=1e-3,rtol=0)
            self.assertLess(r.last_rollout['max_balance_residual_m3'],1e-3)
        state = [{k:v.clone() for k,v in a.actor.state_dict().items()} for a in r.actor]
        a = evaluate_policy(r.actor,self.data,self.physical,2024,1,0,0,diagnostics=True)
        b = evaluate_policy(r.actor,self.data,self.physical,2024,1,0,0,diagnostics=True)
        self.assertEqual(a,b)
        for actor,weights in zip(r.actor,state):
            for key,value in weights.items():
                torch.testing.assert_close(actor.actor.state_dict()[key],value,rtol=0,atol=0)
        self.assertTrue(all(0<=v<=1 for v in a['mean_u']))


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
        _,_,rewards,done,_,_=env.step(np.full(5,199))
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
            closed.step(np.full(5,199));free.step(np.full(5,199))
        np.testing.assert_allclose(closed.storage,initial,atol=1e-4,rtol=0)
        self.assertEqual(closed.last_transition['reward'],0.)
        self.assertLess(free.storage.sum(),initial.sum())

    def test_unreachable_target_raises_without_mutating_environment(self):
        env=self.env(external=np.zeros((2,5)))
        env.storage[0]-=86400.;before=env.storage.copy();index=env.index
        with self.assertRaisesRegex(InfeasibleCascade,'Terminal storage target is unreachable'):
            env.step(np.full(5,199))
        np.testing.assert_array_equal(env.storage,before)
        self.assertEqual(env.index,index);self.assertIsNone(env.last_transition)

    def test_annual_target_cannot_be_silently_applied_to_multiple_years(self):
        env=self.env()
        _,config,_=fixture()
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
            lp,_,_=actor.evaluate_actions(buffer.obs[:-1].reshape(-1,18),
                buffer.actions.reshape(-1,1))
            active=buffer.active_masks[:-1].reshape(-1)>0
            self.assertTrue(active.any());self.assertFalse(active.all())
            np.testing.assert_allclose(lp.detach().numpy()[active],buffer.action_log_probs.reshape(-1,1)[active],atol=1e-6)
            np.testing.assert_array_equal(buffer.action_log_probs.reshape(-1,1)[~active],0.)
            np.testing.assert_array_equal(buffer.actions.reshape(-1,1)[~active],0)
        runner.prep_training();runner.train()
        for buffer in runner.actor_buffer:
            self.assertEqual(float(buffer.factor[0,0,0]),1.)


if __name__=='__main__':
    unittest.main()
