"""Soft-cost arithmetic, unchanged feasibility, and end-to-end reward use."""
import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
import unittest
from unittest.mock import patch
import numpy as np
import torch

from examples.train_cascade import main
from cascade_test_utils import load_configuration
import test_seasonal_operation as seasonal_tests
from harl.envs.cascade_reservoir.data_loader import IDS, OPERATION_PERIODS
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.reward import OperationReward, EpisodeRewardMetrics, reward_settings
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, reference_summary
from harl.envs.cascade_reservoir.diagnostics import before_update, after_update, print_update_window, validation_report
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

DT = seasonal_tests.DT


class RewardPenaltyTests(unittest.TestCase):
    def fixture(self):
        data,config,seasonal=seasonal_tests.SeasonalExecutionTests().fixture()
        config['reward_penalties']=dict(enabled=True,target_band_weight=.1,daily_rise_weight=.1,daily_fall_weight=.1)
        target=seasonal.reference['target_end_of_day_level_m']
        seasonal.reference.update(soft_target_lower_level_m=np.maximum(0.,target-.2),
                                 soft_target_upper_level_m=np.minimum(data.levels(data.vmax),target+.2))
        seasonal.period_parameters={(name,period):dict(suggested_soft_daily_rise_limit_m=.2,
            suggested_soft_daily_fall_limit_m=.3) for name in IDS for period in OPERATION_PERIODS}
        data.training_years=(1965,);data.validation_years=(1965,);data.test_years=();data.test_labels=()
        return data,config,seasonal

    def test_exact_penalty_arithmetic_normalization_and_negative_reward(self):
        data,config,seasonal=self.fixture()
        config['reward_penalties'].update(target_band_weight=2.,daily_rise_weight=3.,daily_fall_weight=4.)
        reward=OperationReward(data,seasonal,reward_settings(config))
        row=dict(storage_start=np.array([7.,.5,.8,.4,.5])*DT,
                 levels_end=np.array([8.,.8,.2,.4,1.]),reward=.7)
        reward.apply(row,0)
        for key,expected in dict(target_band=[1.8,0.,.6,.4,0.],
                                 daily_rise=[.8,.1,0.,0.,.3],daily_fall=[0.,0.,.3,0.,0.]).items():
            np.testing.assert_allclose(row['soft_violations_m'][key],expected,rtol=0,atol=1e-14)
        self.assertAlmostEqual(row['reward_penalty_components']['target_band'],.472)
        self.assertAlmostEqual(row['reward_penalty_components']['daily_rise'],.288)
        self.assertAlmostEqual(row['reward_penalty_components']['daily_fall'],.24)
        self.assertAlmostEqual(row['reward_penalty'],1.)
        self.assertAlmostEqual(row['reward'],-.3)
        self.assertEqual(row['generation_reward'],.7)
        self.assertAlmostEqual(row['reward_penalty_by_reservoir'].sum(),1.)
        metrics=EpisodeRewardMetrics();metrics.update(row)
        totals=metrics.summary(detailed=True)
        self.assertAlmostEqual(totals['episode_generation_reward']-totals['episode_penalty'],totals['episode_reward'])
        self.assertEqual(totals['soft_violation_days_by_reservoir']['daily_fall'],[0,0,1,0,0])

    def test_inside_soft_band_and_rate_limits_has_zero_penalty(self):
        data,config,seasonal=self.fixture();target=seasonal.reference['target_end_of_day_level_m'][0]
        row=dict(storage_start=target*DT,levels_end=target.copy(),reward=.7)
        OperationReward(data,seasonal,reward_settings(config)).apply(row,0)
        self.assertEqual(row['reward'],.7);self.assertEqual(row['reward_penalty'],0.)

    def test_disabled_penalties_preserve_transition_and_zero_weights_preserve_reward(self):
        data,config,_=self.fixture()
        disabled=copy.deepcopy(config);disabled['reward_penalties']['enabled']=False
        zero=copy.deepcopy(config)
        for key in ('target_band_weight','daily_rise_weight','daily_fall_weight'):zero['reward_penalties'][key]=0.
        a,b,c=[CascadeReservoirEnv(data,settings) for settings in (config,disabled,zero)]
        cost=0.
        for _ in range(5):
            for env in (a,b,c):env.step(np.zeros(5))
            for key in ('storage_start','storage_end','release','energy','controls'):
                np.testing.assert_array_equal(a.last_transition[key],b.last_transition[key])
                np.testing.assert_array_equal(a.last_transition[key],c.last_transition[key])
            self.assertEqual(c.last_transition['reward'],b.last_transition['reward'])
            self.assertEqual(a.last_transition['generation_reward'],b.last_transition['reward'])
            cost+=a.last_transition['reward_penalty']
        self.assertGreater(cost,0.)
        np.testing.assert_allclose(a.storage,data.vmax,rtol=0,atol=1e-7)

    def test_bad_settings_and_missing_seasonality_are_rejected(self):
        _,config,_=self.fixture()
        for value in (-1.,np.nan,np.inf,True,None,'invalid',10**400):
            bad=copy.deepcopy(config);bad['reward_penalties']['daily_rise_weight']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite and nonnegative'):
                reward_settings(bad)
        for raw in (None,[],True,dict(enabled='true'),dict(unknown=1)):
            bad=copy.deepcopy(config);bad['reward_penalties']=raw
            with self.subTest(raw=raw),self.assertRaises(ValueError):reward_settings(bad)
        with self.assertRaisesRegex(ValueError,'require seasonal_constraints'):
            reward_settings(dict(config,seasonal_constraints=False))

    def test_hard_infeasibility_still_raises_without_punishing_and_continuing(self):
        data,config,_=self.fixture();data.external[:]=0.
        env=CascadeReservoirEnv(data,config);initial=env.storage.copy()
        with self.assertRaises(InfeasibleCascade):env.step(np.zeros(5))
        np.testing.assert_array_equal(initial,env.storage);self.assertIsNone(env.last_transition)

    def test_policy_and_reference_have_identical_cost_accounting_for_replayed_actions(self):
        data,config,seasonal=self.fixture();env=CascadeReservoirEnv(data,config);actions=[]
        totals=EpisodeRewardMetrics()
        while env.index<env.stop:
            env.step_towards_storage(seasonal.reference['target_end_of_day_storage_m3'][env.index])
            actions.append(env.last_transition['actions'].copy());totals.update(env.last_transition)
        class ReplayActor:
            args={'action_bins':200}
            def __init__(self,agent):self.agent=agent;self.day=0
            def prep_rollout(self):pass
            def act(self,obs,deterministic):
                result=torch.tensor([[actions[self.day][self.agent]]]);self.day+=1;return result
        row=evaluate_policy([ReplayActor(i) for i in range(5)],data,config,1965,1,0,0)
        reference=reference_summary(data,ReservoirPhysics(data,config),[1965])[0]
        for key,value in totals.summary(detailed=True).items():
            self.assertEqual(row[key],value);self.assertEqual(reference[key],value)
        self.assertEqual(row['reward_penalties'],reference['reward_penalties'])
        self.assertAlmostEqual(row['episode_generation_reward'],row['energy_MWh']/env.physics.reference_energy)
        self.assertGreater(row['episode_penalty'],0.)
        self.assertEqual(row['violation_count'],0)  # Soft deviations are separate from hard violations.

    def test_penalties_reach_gae_training_and_diagnostics_in_reward_units(self):
        data,physical,_=self.fixture();cfg,_=load_configuration()
        cfg['algo'].update(gamma=1.,gae_lambda=1.)
        r=CascadeReservoirRunner(data,physical,cfg,17);r.collect_episode(1965)
        totals=r.last_reward_metrics;cb=r.critic_buffer
        self.assertGreater(totals['episode_penalty'],0.)
        self.assertAlmostEqual(float(cb.rewards.astype(float).sum()),totals['episode_reward'],places=7)
        np.testing.assert_allclose(cb.returns[:-1],np.cumsum(cb.rewards[::-1],axis=0)[::-1],rtol=0,atol=1e-6)
        before=before_update(r);r.prep_training();actors,critic=r.train()
        report=after_update(r,before)
        self.assertNotIn('predicted_start_return_TWh',report)
        self.assertEqual(report['realized_sampled_return_reward_units'],report['discounted_episode_reward'])
        row=dict(r.last_rollout,**totals)
        for i,metrics in enumerate(actors):row.update({'actor{}_{}'.format(i,k):float(v) for k,v in metrics.items()})
        row.update({'critic_'+k:float(v) for k,v in critic.items()})
        self.assertTrue(all(np.isfinite(v) for v in row.values()))
        r.training_history=[row];r.diagnostics_history=[report]
        output=io.StringIO()
        with redirect_stdout(output):print_update_window(r,1)
        summary=json.loads(output.getvalue().split('DIAG_UPDATE ',1)[1])
        self.assertNotIn('predicted_start_return_TWh',summary)
        self.assertEqual(summary['last_episode_penalty'],totals['episode_penalty'])
        self.assertAlmostEqual(sum(summary['last_episode_penalty_components'].values()),totals['episode_penalty'])
        self.assertEqual(summary['last_episode_generation_reward'],totals['episode_generation_reward'])

    def test_stochastic_validation_reports_reward_and_preserves_random_state(self):
        data,physical,_=self.fixture();cfg,_=load_configuration()
        runner=CascadeReservoirRunner(data,physical,cfg,17)
        rows=[evaluate_policy(runner.actor,data,physical,1965,17,0,0)]
        rng=torch.random.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):report=validation_report(runner,rows)
        torch.testing.assert_close(rng,torch.random.get_rng_state(),rtol=0,atol=0)
        self.assertEqual(report['deterministic_mean_reward'],rows[0]['episode_reward'])
        self.assertEqual(len(report['stochastic_repeat_means_reward']),3)
        self.assertAlmostEqual(report['stochastic_mean_reward'],np.mean(report['stochastic_repeat_means_reward']))
        self.assertAlmostEqual(report['stochastic_mean_TWh'],np.mean(report['stochastic_repeat_means_TWh']))

    def test_model_selection_uses_reward_while_generation_remains_in_twh(self):
        data,physical,_=self.fixture();cfg,_=load_configuration()
        data.validation_years=tuple(range(2020,2025));data.test_years=(1998,);data.test_labels=('wet',)
        r=CascadeReservoirRunner(data,physical,cfg,19)
        def evaluate(*args,**kwargs):
            energy,reward=((200.,10.),(190.,12.),(220.,11.))[r.iteration]
            return dict(year=args[3],seed=19,energy_MWh=energy*1e6,energy_TWh=energy,
                        episode_reward=reward,episode_penalty=1.)
        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy',side_effect=evaluate),redirect_stdout(io.StringIO()):
            r.validate();r.iteration=1
            with torch.no_grad():next(r.actor[0].actor.parameters()).add_(.1)
            selected=copy.deepcopy(r.actor[0].actor.state_dict());r.validate();r.iteration=2
            with torch.no_grad():next(r.actor[0].actor.parameters()).add_(.1)
            r.validate();rows=r.final_test()
        self.assertEqual(r.best_iteration,1);self.assertEqual(r.best_score,12.)
        self.assertEqual(r.best_energy,190e6)
        self.assertEqual([x['mean_TWh'] for x in r.validation_history],[200.,190.,220.])
        for key,value in selected.items():torch.testing.assert_close(r.actor[0].actor.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(rows[0]['typical_label'],'wet')

    def test_cli_rejects_bad_penalties_before_loading_data(self):
        cfg,physical=load_configuration();physical['reward_penalties']['target_band_weight']=-1.
        with patch('sys.argv',['train_cascade.py','--check-data-only']), \
             patch('examples.train_cascade.load_configuration',return_value=(cfg,physical)), \
             patch('examples.train_cascade.CascadeData') as load,redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main()
        self.assertEqual(error.exception.code,2);load.assert_not_called()

    def test_cli_effective_penalty_switches_and_startup_objective(self):
        data,physical,_=self.fixture();cfg,_=load_configuration()
        for flags,enabled,seasonal in (([],True,True),(['--no-reward-penalties'],False,True),
                (['--reward-penalties'],True,True),(['--no-seasonal-constraints'],False,False),
                (['--no-seasonal-constraints','--no-reward-penalties'],False,False)):
            config=copy.deepcopy(physical)
            if flags==['--reward-penalties']:config['reward_penalties']['enabled']=False
            output=io.StringIO()
            with self.subTest(flags=flags),patch('sys.argv',['train_cascade.py','--check-data-only']+flags), \
                 patch('examples.train_cascade.load_configuration',return_value=(copy.deepcopy(cfg),config)), \
                 patch('examples.train_cascade.CascadeData',return_value=data), \
                 patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]), \
                 redirect_stdout(output):main()
            startup,_=json.JSONDecoder().raw_decode(output.getvalue())
            self.assertEqual(startup['reward_penalties']['enabled'],enabled)
            self.assertEqual(startup['seasonal_constraints'],seasonal)
            self.assertEqual(startup['evaluation_metric'],'undiscounted annual '+
                ('penalized return' if enabled else 'generation'))
        with patch('sys.argv',['train_cascade.py','--no-seasonal-constraints','--reward-penalties']), \
             patch('examples.train_cascade.load_configuration',return_value=(cfg,physical)), \
             patch('examples.train_cascade.CascadeData') as load,redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main()
        self.assertEqual(error.exception.code,2);load.assert_not_called()


if __name__=='__main__':unittest.main()
