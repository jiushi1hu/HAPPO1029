"""Reward decomposition, zero-generation ratios, failure logs and no training changes."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

import test_cascade_reward_penalties as reward_tests
from cascade_test_utils import load_configuration
from examples.train_cascade import main
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.training_metrics import episode_record, log_episode
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class RewardScaleMetricsTests(unittest.TestCase):
    def fixture(self):
        data, physical, _ = reward_tests.RewardPenaltyTests().fixture()
        cfg, _ = load_configuration()
        cfg['model']['hidden_sizes'] = [8,8]
        cfg['algo'].update(ppo_epoch=1, critic_epoch=1)
        cfg['experiment'].update(episodes=1, eval_interval=1, seeds=[1])
        return data, physical, cfg

    def test_episode_sums_ratios_and_raw_soft_exceedances(self):
        env = SimpleNamespace(start=0, index=2, stop=2,
            data=SimpleNamespace(dates=[date(1965,1,1)+timedelta(days=t) for t in range(3)]),
            physics=SimpleNamespace(reference_energy=100.), seasonal=object(),
            reward_settings={'enabled':True}, terminal_storage_target=None,
            observation_metadata=lambda:dict(actor_observation_layout='local_neighbors_v1',
                                             actor_observation_size=16,critic_state_size=43))
        metrics = EpisodeRewardMetrics(1.)
        for power,costs,excess in [(1.,(.1,.2),([1,0,0,0,0],[0,3,0,0,0])),
                                   (2.,(.2,.1),([2,0,1,0,0],[0]*5))]:
            components=dict(zip(('guaranteed_output','ecological_flow'),costs))
            metrics.update(dict(reward=power-sum(costs),generation_reward=power,
                reward_penalty=sum(costs),reward_penalty_components=components,
                service_shortfalls={k:np.array(v,dtype=float) for k,v in zip(components,excess)},
                service_active_masks={k:np.ones(5,dtype=bool) for k in components},
                service_shortfall_integrals={k:np.array(v,dtype=float)*(24. if k=='guaranteed_output' else 86400.)
                                             for k,v in zip(components,excess)}))
        row=episode_record(env,metrics,np.full(5,60.),1,1,1.,.95,0)
        self.assertEqual(row['actual_generation_MWh'],300.)
        self.assertEqual(row['power_reward'],3.)
        self.assertAlmostEqual(row['total_reward'],2.4)
        self.assertAlmostEqual(row['penalty_to_power_percent']['guaranteed_output'],10.)
        self.assertAlmostEqual(row['penalty_to_power_percent']['total'],20.)
        band=row['service_shortfalls']['guaranteed_output']
        self.assertEqual(band['reservoir_day_count'],3)
        self.assertEqual(band['reservoir_day_rate'],.3)
        self.assertEqual(band['days_any_reservoir'],2)
        self.assertEqual(band['shortfall_energy_MWh'],96.)
        self.assertEqual(band['max_shortfall'],2.)
        self.assertEqual(band['by_reservoir_days'],[2,0,1,0,0])
        self.assertEqual(row['hard_constraint_failure_rate'],0.)
        self.assertEqual(row['actor_observation_layout'],'local_neighbors_v1')
        self.assertEqual(row['actor_observation_size'],16)
        self.assertEqual(row['critic_state_size'],43)
        stream,console=io.StringIO(),io.StringIO()
        with redirect_stdout(console):log_episode(row,stream)
        self.assertEqual(json.loads(stream.getvalue()),row)
        self.assertEqual(json.loads(console.getvalue().split(' ',1)[1]),row)

    def test_zero_generation_and_first_day_failure_are_explicit_and_json_safe(self):
        data,physical,cfg=self.fixture(); data.external[:]=0.
        stream,console=io.StringIO(),io.StringIO()
        runner=CascadeReservoirRunner(data,physical,cfg,1,episode_log=stream)
        with redirect_stdout(console), self.assertRaises(InfeasibleCascade):runner.collect_episode(1965)
        row=json.loads(stream.getvalue())
        self.assertFalse(row['completed'])
        self.assertEqual(row['simulated_days'],0)
        self.assertEqual(row['actual_generation_MWh'],0.)
        self.assertIsNone(row['annual_generation_TWh'])
        self.assertTrue(all(x is None for x in row['penalty_to_power_percent'].values()))
        self.assertEqual(row['hard_constraint_failure_rate'],1.)
        self.assertEqual(row['hard_constraint_failure_rate_so_far'],1.)
        self.assertEqual(row['failure']['error_type'],'InfeasibleCascade')
        self.assertEqual(runner.training_history,[])
        self.assertEqual(runner.completed_episodes,0)

    def test_later_failure_keeps_only_executed_generation_and_reraises(self):
        data,physical,cfg=self.fixture()
        runner=CascadeReservoirRunner(data,physical,cfg,1)
        with redirect_stdout(io.StringIO()):runner.collect_episode(1965)
        original=CascadeReservoirEnv.step
        def step(env,actions):
            if env.index>env.start:raise InfeasibleCascade('test deadline unreachable')
            return original(env,actions)
        with patch.object(CascadeReservoirEnv,'step',step), redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(InfeasibleCascade,'test deadline unreachable'):
            runner.collect_episode(1965)
        row=runner.episode_history[-1]
        self.assertEqual(row['attempted_episodes'],2)
        self.assertEqual(row['failed_episodes_so_far'],1)
        self.assertEqual(row['hard_constraint_failure_rate_so_far'],.5)
        self.assertEqual(row['simulated_days'],1)
        self.assertAlmostEqual(row['actual_generation_MWh'],runner.envs[0].last_transition['energy'].sum())
        self.assertIsNone(row['annual_generation_TWh'])
        self.assertEqual(row['total_reward'],runner.envs[0].last_transition['reward'])

    def test_penalties_disabled_still_measure_excess_without_changing_rewards(self):
        data,physical,cfg=self.fixture();physical['reward_penalties']['enabled']=False
        runner=CascadeReservoirRunner(data,physical,cfg,1)
        with redirect_stdout(io.StringIO()):runner.collect_episode(1965)
        row=runner.episode_history[0]
        self.assertEqual(row['power_reward'],row['total_reward'])
        self.assertEqual(row['total_soft_penalty'],0.)
        self.assertGreater(sum(v['reservoir_day_count'] for v in row['service_shortfalls'].values()),0)
        self.assertTrue(all(v==0. for v in row['penalty_to_power_percent'].values()))

    def test_logging_preserves_buffers_rng_and_parameter_updates(self):
        outcomes=[]
        for logging in (False,True):
            data,physical,cfg=self.fixture()
            runner=CascadeReservoirRunner(data,physical,cfg,1,episode_log=io.StringIO() if logging else None)
            with patch.object(runner,'_log_episode',wraps=runner._log_episode) as log, redirect_stdout(io.StringIO()):
                if not logging:log.side_effect=lambda *args,**kwargs:None
                runner.collect_episode(1965)
            actions=[b.actions.copy() for b in runner.actor_buffer]
            rewards=runner.critic_buffer.rewards.copy()
            runner.prep_training();runner.train()
            outcomes.append((actions,rewards,copy.deepcopy(runner.actor[0].actor.state_dict()),
                             copy.deepcopy(runner.critic.critic.state_dict()),torch.get_rng_state().clone()))
        a,b=outcomes
        for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(a[1],b[1])
        for x,y in zip(a[2:4],b[2:4]):
            for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        torch.testing.assert_close(a[4],b[4],rtol=0,atol=0)

    def test_cli_jsonl_writes_each_episode_and_refuses_overwrite(self):
        data,physical,cfg=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'logs'/'episodes.jsonl'
            argv=['train_cascade.py','--episodes','1','--eval-interval','1','--seeds','1',
                  '--no-plot','--skip-final-test','--episode-log',str(path)]
            with patch('sys.argv',argv), \
                 patch('examples.train_cascade.load_configuration',return_value=(cfg,physical)), \
                 patch('examples.train_cascade.CascadeData',return_value=data), \
                 patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]), \
                 patch('examples.train_cascade.reported_references',return_value=[dict(year=1965,energy_TWh=1.)]), \
                 redirect_stdout(io.StringIO()):main()
            lines=path.read_text().splitlines()
            self.assertEqual(len(lines),1)
            row=json.loads(lines[0]);self.assertEqual(row['episode'],1)
            self.assertTrue(row['completed'])
            with patch('sys.argv',argv), patch('examples.train_cascade.CascadeData') as load, \
                 redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
            self.assertEqual(path.read_text().splitlines(),lines)
            load.assert_not_called()


if __name__=='__main__':unittest.main()
