"""One reward objective across GAE, logs, policy evaluation and references."""
import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch

import test_reward_scale_metrics as fixtures
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.evaluation import evaluate_policy, reference_summary
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics, PENALTY_TERMS
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class DiscountConsistencyTests(unittest.TestCase):
    def test_components_use_the_same_discount_and_counts_remain_physical(self):
        for gamma in (0., .5, .99, 1.):
            with self.subTest(gamma=gamma):
                metrics = EpisodeRewardMetrics(gamma)
                for power, costs in ((10., (1., 2.)), (20., (2., 3.))):
                    metrics.update(dict(reward=power-sum(costs), generation_reward=power,
                        reward_penalty=sum(costs), reward_penalty_components=dict(zip(PENALTY_TERMS, costs)),
                        reward_penalty_by_reservoir=np.full(5, sum(costs)/5),
                        service_shortfalls={term: np.ones(5) for term in PENALTY_TERMS},
                        service_active_masks={term: np.ones(5, dtype=bool) for term in PENALTY_TERMS}))
                row = metrics.summary(detailed=True)
                self.assertAlmostEqual(row['episode_generation_reward'], 10.+gamma*20.)
                self.assertAlmostEqual(row['episode_penalty'], 3.+gamma*5.)
                self.assertAlmostEqual(row['episode_reward'], 7.+gamma*15.)
                self.assertAlmostEqual(sum(row['penalty_by_reservoir']), row['episode_penalty'])
                self.assertEqual(row['service_shortfalls']['ecological_flow']['by_reservoir_days'], [2]*5)
                self.assertEqual(metrics.undiscounted_generation_reward, 30.)

    def test_rollout_logging_and_lambda_one_gae_have_the_same_return(self):
        for gamma in (0., .5, .99, 1.):
            with self.subTest(gamma=gamma):
                data, physical, cfg = fixtures.RewardScaleMetricsTests().fixture()
                cfg['algo'].update(gamma=gamma, gae_lambda=1.)
                runner = CascadeReservoirRunner(data, physical, cfg, 2)
                with redirect_stdout(io.StringIO()):
                    runner.collect_episode(1965)
                daily = runner.critic_buffer.rewards[:, 0, 0].astype(float)
                expected = float(np.dot(gamma**np.arange(len(daily)), daily))
                self.assertAlmostEqual(runner.last_reward_metrics['episode_reward'], expected, places=7)
                self.assertAlmostEqual(runner.episode_history[-1]['total_reward'], expected, places=7)
                self.assertAlmostEqual(float(runner.critic_buffer.returns[0, 0, 0]), expected, places=6)
                self.assertEqual(runner.episode_history[-1]['gamma'], gamma)

    def test_policy_and_reference_discount_the_same_trajectory_and_keep_energy(self):
        data, physical, _ = fixtures.RewardScaleMetricsTests().fixture()
        physics = ReservoirPhysics(data, physical)
        env = CascadeReservoirEnv(data, physical)
        actions, daily_rewards, daily_generation = [], [], []
        while env.index < env.stop:
            env.step_towards_storage(env.seasonal.reference['target_end_of_day_storage_m3'][env.index])
            row = env.last_transition
            actions.append(row['actions'].copy())
            daily_rewards.append(row['reward'])
            daily_generation.append(float(row['energy'].sum()))

        class ReplayActor:
            args = {'action_bins': 200}
            def __init__(self, agent):
                self.agent, self.day = agent, 0
            def prep_rollout(self):
                pass
            def act(self, obs, deterministic):
                action = torch.tensor([[actions[self.day][self.agent]]])
                self.day += 1
                return action

        for gamma in (0., .5, .99, 1.):
            with self.subTest(gamma=gamma):
                row = evaluate_policy([ReplayActor(i) for i in range(5)], data, physical,
                                      1965, 1, 0, 0, gamma=gamma)
                reference = reference_summary(data, physics, [1965], gamma=gamma)[0]
                expected = float(np.dot(gamma**np.arange(len(daily_rewards)), daily_rewards))
                self.assertAlmostEqual(row['episode_reward'], expected)
                self.assertAlmostEqual(row['episode_reward'], reference['episode_reward'])
                self.assertAlmostEqual(row['episode_generation_reward'], reference['episode_generation_reward'])
                self.assertAlmostEqual(row['episode_penalty'], reference['episode_penalty'])
                self.assertAlmostEqual(row['energy_MWh'], sum(daily_generation))
                self.assertEqual(row['gamma'], gamma)

    def test_selection_uses_discounted_reward_even_without_penalties(self):
        for enabled in (False, True):
            with self.subTest(penalties=enabled):
                data, physical, cfg = fixtures.RewardScaleMetricsTests().fixture()
                physical['reward_penalties']['enabled'] = enabled
                cfg['algo']['gamma'] = .5
                data.validation_years = tuple(range(2020, 2025))
                runner = CascadeReservoirRunner(data, physical, cfg, 3)

                def evaluate(actors, data, physical, year, seed, episode, steps, *, gamma):
                    self.assertEqual(gamma, .5)
                    daily_energy = np.array([100., 0.]) if episode == 0 else np.array([0., 150.])
                    return dict(year=year, energy_MWh=float(daily_energy.sum()),
                        energy_TWh=float(daily_energy.sum()/1e6), episode_penalty=0.,
                        episode_reward=float(np.dot(gamma**np.arange(2), daily_energy)/1200000.))

                with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=evaluate), \
                        redirect_stdout(io.StringIO()):
                    runner.validate()
                    runner.iteration = 1
                    runner.validate()
                self.assertEqual(runner.best_iteration, 0)
                self.assertEqual(runner.best_energy, 100.)
                self.assertEqual([row['mean_TWh'] for row in runner.validation_history], [100.e-6, 150.e-6])
                self.assertGreater(runner.validation_history[0]['mean_reward'],
                                   runner.validation_history[1]['mean_reward'])

    def test_constant_storage_reference_uses_gamma_in_generation_only_mode(self):
        data, physical, _ = fixtures.RewardScaleMetricsTests().fixture()
        physical = copy.deepcopy(physical)
        physical.update(seasonal_constraints=False, reward_penalties={'enabled': False})
        physics = ReservoirPhysics(data, physical)
        start, stop = data.indices(1965, 1965)
        _, daily = physics.constant_storage(data.external[start:stop], data.initial_storage(1965))
        gamma = .5
        row = reference_summary(data, physics, [1965], gamma=gamma)[0]
        expected = np.dot(gamma**np.arange(len(daily)), daily.sum(axis=1)/physics.reference_energy)
        self.assertAlmostEqual(row['episode_reward'], float(expected))
        self.assertAlmostEqual(row['energy_MWh'], float(daily.sum()))

    def test_fixed_year_evaluation_uses_discounted_components_for_selection(self):
        for enabled in (False, True):
            with self.subTest(penalties=enabled):
                data, physical, cfg = fixtures.RewardScaleMetricsTests().fixture()
                physical['reward_penalties']['enabled'] = enabled
                cfg['algo']['gamma'] = .5
                cfg['experiment']['fixed_training_year'] = 1965
                runner = CascadeReservoirRunner(data, physical, cfg, 4)
                cost = .01 if enabled else 0.
                row = dict(energy_MWh=1000000., energy_TWh=1., episode_generation_reward=.1,
                    episode_reward=.1-cost, episode_penalty_guaranteed_output=cost,
                    episode_penalty_ecological_flow=0.,
                    feasible=True, violation_count=0)
                with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', return_value=row) as evaluate, \
                        redirect_stdout(io.StringIO()):
                    score = runner.evaluate_training_year()
                self.assertEqual(evaluate.call_args.kwargs['gamma'], .5)
                self.assertEqual(score, .1-cost)
                self.assertEqual(runner.best_score, score)
                self.assertEqual(runner.fixed_year_history[0]['power_reward'], .1)

    def test_invalid_or_missing_gamma_cannot_silently_use_another_objective(self):
        for gamma in (-1., 1.1, np.nan, np.inf, True, None):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                EpisodeRewardMetrics(gamma)
        with self.assertRaises(TypeError):
            EpisodeRewardMetrics()
        with self.assertRaises(TypeError):
            evaluate_policy([], None, None, 1965, 1, 0, 0)
        with self.assertRaises(TypeError):
            reference_summary(None, None, [1965])


if __name__ == '__main__':
    unittest.main()
