"""Frozen-policy multi-year batches, independent GAE, and exact episode budgets."""
import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cascade_test_utils import load_configuration
from examples.train_cascade import main
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.diagnostics import before_update, monte_carlo_returns
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner, validate_rollout_episodes
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
import test_cascade_reward_penalties as reward_fixtures

ROOT=Path(__file__).resolve().parents[1]


class RolloutBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.physical=load_configuration()
        cls.cfg['model']['hidden_sizes']=[8,8]
        cls.cfg['algo'].update(ppo_epoch=1,critic_epoch=1)
        cls.cfg['experiment']['rollout_episodes']=6
        cls.data=CascadeData(ROOT/'data/cascade_reservoir_new',cls.physical,cls.cfg['experiment'])

    def test_mixed_common_and_leap_years_preserve_all_rows_and_independent_returns(self):
        r=CascadeReservoirRunner(self.data,self.physical,copy.deepcopy(self.cfg),7)
        snapshots=[]
        original=r.collect_episode
        before=[copy.deepcopy(a.actor.state_dict()) for a in r.actor]
        critic_before=copy.deepcopy(r.critic.critic.state_dict())
        norm_before=copy.deepcopy(r.value_normalizer.state_dict())

        def collect():
            original()
            snapshots.append((r.actor_buffer,r.critic_buffer,dict(r.last_reward_metrics)))

        with patch.object(r,'sample_training_year',side_effect=[1965,1968,1965]), \
             patch.object(r,'collect_episode',side_effect=collect),redirect_stdout(io.StringIO()):
            r.collect_batch(3)
        self.assertEqual(r.batch_episode_lengths,[365,366,365])
        self.assertEqual(r.batch_years,[1965,1968,1965])
        self.assertEqual(r.length,1096)
        self.assertEqual(r.completed_episodes,3)
        np.testing.assert_array_equal(r.batch_episode_starts,[0,365,731])
        np.testing.assert_array_equal(r.critic_buffer.masks[[365,731,1096]],0.)
        for i,target in enumerate(r.actor_buffer):
            np.testing.assert_array_equal(target.obs[:-1],np.concatenate([s[0][i].obs[:-1] for s in snapshots]))
            np.testing.assert_array_equal(target.actions,np.concatenate([s[0][i].actions for s in snapshots]))
            np.testing.assert_array_equal(target.active_masks[:-1],np.concatenate([s[0][i].active_masks[:-1] for s in snapshots]))
            np.testing.assert_array_equal(target.action_log_probs,np.concatenate([s[0][i].action_log_probs for s in snapshots]))
            np.testing.assert_allclose(target.discount_weights(.99)[:,0,0],
                np.concatenate([.99**np.arange(n) for n in (365,366,365)]),rtol=6e-8)
            for name,value in before[i].items():
                torch.testing.assert_close(value,r.actor[i].actor.state_dict()[name],rtol=0,atol=0)
        for name,value in critic_before.items():
            torch.testing.assert_close(value,r.critic.critic.state_dict()[name],rtol=0,atol=0)
        for name,value in norm_before.items():
            torch.testing.assert_close(value,r.value_normalizer.state_dict()[name],rtol=0,atol=0)
        expected=np.concatenate([s[1].returns[:-1] for s in snapshots])
        np.testing.assert_array_equal(r.critic_buffer.returns[:-1],expected)
        cb=r.critic_buffer
        expected_mc=np.concatenate([monte_carlo_returns(s[1].rewards,s[1].masks,.99) for s in snapshots])
        np.testing.assert_array_equal(monte_carlo_returns(cb.rewards,cb.masks,.99),expected_mc)
        cb.compute_returns(np.full((1,1),12345.,dtype=np.float32),r.value_normalizer)
        np.testing.assert_allclose(cb.returns[:-1],expected,rtol=0,atol=1e-6)
        self.assertEqual(r.last_reward_metrics['gamma'],.99)
        self.assertAlmostEqual(r.last_reward_metrics['episode_reward'],
                               np.mean([s[2]['episode_reward'] for s in snapshots]))
        # The real sequential HAPPO update must accept every concatenated row.
        snapshot=before_update(r)
        self.assertEqual(snapshot['mc'].shape,(1096,1,1))
        r.prep_training()
        actors,critic=r.train()
        self.assertTrue(all(np.isfinite(float(v)) for info in actors+[critic] for v in info.values()))

    def test_six_plus_one_budget_updates_only_after_collection_and_evaluates_each_round(self):
        data,physical,_=reward_fixtures.RewardPenaltyTests().fixture()
        cfg=copy.deepcopy(self.cfg)
        cfg['experiment'].update(fixed_training_year=1965,episodes=7,diagnostics=True)
        r=CascadeReservoirRunner(data,physical,cfg,2,episode_log=io.StringIO())
        seen=[]
        original=r.collect_episode
        def collect():
            seen.append(r.update_count)
            original()
        with patch.object(r,'collect_episode',side_effect=collect),redirect_stdout(io.StringIO()):
            result=r.run(7,1)
        self.assertEqual(seen,[0]*6+[1])
        self.assertEqual(r.completed_episodes,7)
        self.assertEqual(r.update_count,2)
        self.assertEqual(r.training_steps,35)
        self.assertEqual([x['episode'] for x in r.fixed_year_history],[0,6,7])
        self.assertEqual([x['update'] for x in r.fixed_year_history],[0,1,2])
        self.assertEqual([x['iteration'] for x in r.training_history],[1,2])
        self.assertEqual([x['batch_episodes'] for x in r.training_history],[6,1])
        self.assertEqual(result['updates_completed'],2)
        self.assertEqual(len(r.episode_history),7)
        for record,policy_update in zip(r.episode_history,seen):
            self.assertEqual(record['policy_updates_completed'],policy_update)
        for update in r.diagnostics_history:
            self.assertAlmostEqual(update['episode_reward'],
                r.training_history[update['update']-1]['episode_reward'],places=6)
        records=[json.loads(line) for line in r.episode_log.getvalue().splitlines()]
        self.assertEqual(len([x for x in records if x['record_type']=='training_episode']),7)
        self.assertEqual(len([x for x in records if x['record_type']=='training_update']),2)

    def test_validation_history_contains_frozen_per_year_results(self):
        data,physical,_=reward_fixtures.RewardPenaltyTests().fixture()
        cfg=copy.deepcopy(self.cfg)
        cfg['experiment'].update(episodes=2,diagnostics=False)
        r=CascadeReservoirRunner(data,physical,cfg,3,episode_log=io.StringIO())
        with redirect_stdout(io.StringIO()):r.run(2,1)
        rows=[json.loads(line) for line in r.episode_log.getvalue().splitlines()]
        validation=[x for x in rows if x['record_type']=='validation_checkpoint']
        self.assertEqual([x['episode'] for x in validation],[0,2])
        self.assertTrue(all('service_shortfalls' in year for row in validation for year in row['years']))

    def test_batch_size_validation_and_cli_reject_before_loading_data(self):
        for value in (0,-1,True,1.5,'6',None):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'positive integer'):
                validate_rollout_episodes(value)
        for value in ('0','-1'):
            with patch('sys.argv',['train_cascade.py','--check-data-only','--rollout-episodes',value]), \
                 patch('examples.train_cascade.CascadeData') as data,redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):main()
            data.assert_not_called()

    def test_cli_reports_annual_budget_and_update_count_separately(self):
        output=io.StringIO()
        with patch('sys.argv',['train_cascade.py','--check-data-only','--episodes','7',
                               '--rollout-episodes','6','--eval-interval','1']), \
             patch('examples.train_cascade.load_configuration',return_value=(copy.deepcopy(self.cfg),self.physical)), \
             patch('examples.train_cascade.CascadeData',return_value=self.data), \
             patch('examples.train_cascade.validate_operating_feasibility',return_value=[]), \
             redirect_stdout(output):main()
        metadata,_=json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(metadata['episodes_per_seed'],7)
        self.assertEqual(metadata['rollout_episodes_per_update'],6)
        self.assertEqual(metadata['updates_per_seed'],2)
        self.assertEqual(metadata['eval_interval_updates'],1)
        self.assertEqual(metadata['gamma'],self.cfg['algo']['gamma'])
        self.assertEqual(metadata['reward_penalties'],self.physical['reward_penalties'])

    def test_diagnostic_milestones_run_at_the_first_completed_batch_after_crossing(self):
        data,physical,_=reward_fixtures.RewardPenaltyTests().fixture()
        cfg=copy.deepcopy(self.cfg)
        cfg['experiment'].update(episodes=300,diagnostics=True,diagnostic_eval_episodes=[20])
        r=CascadeReservoirRunner(data,physical,cfg,4)
        with patch('harl.envs.cascade_reservoir.diagnostics.validation_report',return_value={}) as report, \
             redirect_stdout(io.StringIO()):
            for episode in (18,24,30):
                r.iteration=episode
                r.validate()
        self.assertEqual(report.call_count,1)


if __name__=='__main__':unittest.main()
