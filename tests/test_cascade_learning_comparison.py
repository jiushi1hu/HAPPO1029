"""Learning-signal comparisons keep year streams, evaluation and budgets explicit."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

import examples.compare_cascade_learning_signal as comparison
import test_reward_scale_metrics as fixtures


class LearningComparisonTests(unittest.TestCase):
    def fixture(self):
        data, physical, cfg = fixtures.RewardScaleMetricsTests().fixture()
        data.validation_years = ()
        data.test_years = ()
        data.hashes = {'fixture': 'five-day-training-only-cascade'}
        cfg['experiment'].update(rollout_episodes=2, fixed_training_year=None)
        return data, physical, cfg

    @staticmethod
    def arm(candidate=False, intervention='gae_lambda', repeats=3):
        rewards = [0., 4., 8.][:repeats] if candidate else [1., 2., 3.][:repeats]
        settings = dict(gae_lambda=1. if candidate and intervention=='gae_lambda' else .95,
                        rollout_episodes=4 if candidate and intervention=='rollout_episodes' else 2)
        updates = 2 if candidate and intervention=='rollout_episodes' else 3
        return dict(seed=7, intervention=intervention, settings=settings,
            sampled_years=[1965]*5, training_steps=25,
            initial_training_state_hash='common',
            final_training_state_hash='candidate' if candidate else 'baseline',
            final_actor_state_hash='candidate-actors' if candidate else 'baseline-actors',
            actor_optimizer_steps=[float(updates)]*5,
            final_policy_report=dict(seed=7,gamma=.99,scope='training_years',
                evaluation_years=[1965],sampled_repeats=repeats,
                evaluation_seeds=[[7000+i] for i in range(repeats)],
                episode=5, update=updates, training_steps=25,
                stochastic_repeat_means_reward=rewards,
                deterministic_mean_reward=4. if candidate else 1.))

    def run_cli(self, arguments):
        data, physical, cfg = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'comparison.jsonl'
            argv = ['compare_cascade_learning_signal.py','--training-years','1965',
                    '--seeds','1','--evaluation-repeats','1','--output',str(output)]+arguments
            with patch('sys.argv',argv), \
                    patch.object(comparison,'load_configuration',return_value=(cfg,physical)), \
                    patch.object(comparison,'CascadeData',return_value=data), \
                    patch.object(comparison,'validate_operating_feasibility',return_value=[1965]) as check, \
                    patch.object(comparison.CascadeReservoirRunner,'validate',side_effect=AssertionError('holdout validation')), \
                    patch.object(comparison.CascadeReservoirRunner,'final_test',side_effect=AssertionError('held-out test')), \
                    redirect_stdout(io.StringIO()):
                comparison.main()
            check.assert_called_once_with(data,physical,years=[1965])
            return [json.loads(line) for line in output.read_text().splitlines()]

    def test_settings_change_exactly_one_factor_without_modifying_input(self):
        _, _, cfg = self.fixture()
        old = copy.deepcopy(cfg)
        baseline, candidate = comparison.comparison_settings(cfg,'gae_lambda')
        self.assertEqual(baseline,dict(gae_lambda=.95,rollout_episodes=2))
        self.assertEqual(candidate,dict(gae_lambda=1.,rollout_episodes=2))
        baseline, candidate = comparison.comparison_settings(cfg,'rollout_episodes',candidate_rollout_episodes=4)
        self.assertEqual(candidate,dict(gae_lambda=.95,rollout_episodes=4))
        self.assertEqual(cfg,old)
        for kind, options in [('gae_lambda',dict(candidate_gae_lambda=.95)),
                              ('gae_lambda',dict(candidate_gae_lambda=float('nan'))),
                              ('gae_lambda',dict(candidate_rollout_episodes=4)),
                              ('rollout_episodes',dict(candidate_gae_lambda=1.)),
                              ('rollout_episodes',dict(candidate_rollout_episodes=2)),
                              ('rollout_episodes',dict(candidate_rollout_episodes=0))]:
            with self.subTest(kind=kind,options=options),self.assertRaises(ValueError):
                comparison.comparison_settings(cfg,kind,**options)
        cfg['algo']['use_gae']=False
        with self.assertRaisesRegex(ValueError,'use_gae'):
            comparison.comparison_settings(cfg,'gae_lambda')

    def test_year_schedule_is_repeatable_and_does_not_consume_global_rng(self):
        np.random.seed(19)
        state = np.random.get_state()
        expected = np.random.RandomState(7).choice([1965,1968],size=20).tolist()
        self.assertEqual(comparison.year_schedule([1965,1968],20,7),expected)
        self.assertEqual(comparison.year_schedule([1965,1968],20,7),expected)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1],after[1])
        self.assertEqual(state[2:],after[2:])
        for years in ([],[1965,1965]):
            with self.assertRaises(ValueError):comparison.year_schedule(years,2,7)

    def test_pairing_uses_repeat_differences_and_allows_batch_update_counts_to_differ(self):
        pair = comparison.paired_learning_result(self.arm(),self.arm(True))
        np.testing.assert_array_equal(pair['sampled_repeat_reward_differences'],[-1.,2.,5.])
        self.assertEqual(pair['sampled_reward_difference_mean'],2.)
        self.assertEqual(pair['sampled_reward_difference_sd'],3.)
        self.assertAlmostEqual(pair['sampled_reward_difference_se'],np.sqrt(3.))
        self.assertTrue(pair['initial_state_match'])
        self.assertTrue(pair['equal_update_budgets'])
        self.assertEqual(pair['deterministic_reward_difference'],3.)
        pair = comparison.paired_learning_result(self.arm(intervention='rollout_episodes'),
            self.arm(True,intervention='rollout_episodes'))
        self.assertEqual((pair['baseline_updates'],pair['candidate_updates']),(3,2))
        self.assertFalse(pair['equal_update_budgets'])

    def test_pairing_rejects_unmatched_grids_year_streams_and_extra_treatments(self):
        for key,value in [('gamma',1.),('scope','validation'),('seed',8),
                          ('evaluation_years',[1968]),('evaluation_seeds',[[1],[2],[3]]),
                          ('episode',6),('training_steps',26),('sampled_repeats',2)]:
            baseline,candidate=self.arm(),self.arm(True)
            candidate['final_policy_report'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                comparison.paired_learning_result(baseline,candidate)
        for change in ('schedule','settings','seed','steps','initial','updates'):
            baseline,candidate=self.arm(),self.arm(True)
            if change=='schedule':candidate['sampled_years'][-1]=1968
            elif change=='settings':candidate['settings']['rollout_episodes']=4
            elif change=='seed':candidate['seed']=8
            elif change=='steps':candidate['training_steps']=26
            elif change=='initial':candidate['initial_training_state_hash']='different'
            else:candidate['final_policy_report']['update']=2
            with self.subTest(change=change),self.assertRaises(ValueError):
                comparison.paired_learning_result(baseline,candidate)

    def test_pairing_rejects_malformed_repeat_arrays_and_does_not_invent_single_repeat_se(self):
        for values in ([],[1.],[[1.],[2.],[3.]],[1.,float('nan'),3.],[1.,2.,float('inf')]):
            baseline,candidate=self.arm(),self.arm(True)
            candidate['final_policy_report']['stochastic_repeat_means_reward']=values
            with self.subTest(values=values),self.assertRaises(ValueError):
                comparison.paired_learning_result(baseline,candidate)
        pair=comparison.paired_learning_result(self.arm(repeats=1),self.arm(True,repeats=1))
        self.assertIsNone(pair['sampled_reward_difference_sd'])
        self.assertIsNone(pair['sampled_reward_difference_se'])

    def test_seed_summary_uses_seed_estimates_rather_than_pooled_days_or_repeats(self):
        pairs=[]
        for seed,difference in enumerate((1.,3.,5.),1):
            pair=comparison.paired_learning_result(self.arm(),self.arm(True))
            pair.update(seed=seed,sampled_reward_difference_mean=difference,
                        deterministic_reward_difference=difference+1.)
            pairs.append(pair)
        result=comparison.learning_comparison_summary(pairs)
        self.assertEqual(result['sampled_reward_difference_across_seeds']['mean'],3.)
        self.assertEqual(result['sampled_reward_difference_across_seeds']['sd'],2.)
        self.assertAlmostEqual(result['sampled_reward_difference_across_seeds']['se'],2./np.sqrt(3.))
        self.assertIsNone(comparison.learning_comparison_summary(pairs[:1])[
            'sampled_reward_difference_across_seeds']['se'])
        with self.assertRaises(ValueError):comparison.learning_comparison_summary([pairs[0],pairs[0]])

    def test_cli_lambda_trial_preserves_annual_and_update_budgets_and_emits_learning_diagnostics(self):
        rows=self.run_cli(['--intervention','gae_lambda','--episodes','4'])
        meta=rows[0]
        self.assertEqual(meta['changed_settings'],['gae_lambda'])
        arms=[row for row in rows if row['record_type']=='learning_signal_result']
        self.assertEqual(len(arms),2)
        self.assertEqual([a['annual_episodes'] for a in arms],[4,4])
        self.assertEqual([a['updates'] for a in arms],[2,2])
        self.assertEqual(arms[0]['sampled_years'],arms[1]['sampled_years'])
        self.assertEqual(arms[0]['initial_training_state_hash'],arms[1]['initial_training_state_hash'])
        for arm in arms:
            self.assertEqual([r['batch_episodes'] for r in arm['training_history']],[2,2])
            self.assertTrue(all('learning_signal' in d for d in arm['training_diagnostics']))
            self.assertEqual(arm['final_policy_report']['scope'],'training_years')
            self.assertTrue(arm['final_policy_report']['evaluation_preserved_training_state'])
        self.assertEqual(rows[-1]['record_type'],'paired_learning_signal_summary')

    def test_cli_batch_trial_retains_partial_final_batch_and_records_fewer_updates(self):
        rows=self.run_cli(['--intervention','rollout_episodes','--episodes','5',
                          '--candidate-rollout-episodes','4'])
        arms=[row for row in rows if row['record_type']=='learning_signal_result']
        self.assertEqual([a['annual_episodes'] for a in arms],[5,5])
        self.assertEqual([a['updates'] for a in arms],[3,2])
        self.assertEqual([r['batch_episodes'] for r in arms[0]['training_history']],[2,2,1])
        self.assertEqual([r['batch_episodes'] for r in arms[1]['training_history']],[4,1])
        self.assertEqual(arms[0]['training_steps'],arms[1]['training_steps'])
        pair=next(row for row in rows if row['record_type']=='paired_learning_signal_comparison')
        self.assertFalse(pair['equal_update_budgets'])
        self.assertTrue(pair['year_schedule_match'])

    def test_endpoint_evaluation_preserves_full_learning_state_and_does_not_visit_holdouts(self):
        data,physical,cfg=self.fixture()
        cfg['experiment']['diagnostic_stochastic_repeats']=1
        runner=comparison.TrainingSignalRunner(data,physical,cfg,7,
            evaluation_years=[1965],annual_episodes=2)
        runner.actor[0].actor.eval()
        before=comparison.training_state_fingerprint(runner)
        with redirect_stdout(io.StringIO()):report=comparison.evaluate_training_endpoint(runner,[1965])
        self.assertEqual(before,comparison.training_state_fingerprint(runner))
        self.assertEqual(report['evaluation_years'],[1965])
        for years in ([1966],[1965,1965]):
            with self.assertRaises(ValueError):comparison.evaluate_training_endpoint(runner,years)
        data.validation_years=(1965,)
        with self.assertRaises(ValueError):comparison.evaluate_training_endpoint(runner,[1965])

    def test_cli_rejects_invalid_settings_and_existing_output_before_data(self):
        for options in (['--episodes','0'],['--evaluation-repeats','0'],['--action-bins','1'],
                        ['--rollout-episodes','0'],['--candidate-gae-lambda','nan'],
                        ['--candidate-gae-lambda','.95'],['--seeds','1','1'],
                        ['--training-years','1965','1965'],['--joint-target-kl','.01']):
            with tempfile.TemporaryDirectory() as directory:
                argv=['compare_cascade_learning_signal.py','--intervention','gae_lambda',
                      '--output',str(Path(directory)/'new.jsonl')]+options
                with patch('sys.argv',argv),patch.object(comparison,'CascadeData') as data, \
                        redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as failure:
                    comparison.main()
                self.assertEqual(failure.exception.code,2)
                data.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'existing.jsonl';output.write_text('preserve\n')
            with patch('sys.argv',['compare_cascade_learning_signal.py','--intervention','gae_lambda',
                                   '--output',str(output)]),patch.object(comparison,'CascadeData') as data, \
                    redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):comparison.main()
            data.assert_not_called()
            self.assertEqual(output.read_text(),'preserve\n')


if __name__=='__main__':unittest.main()
