"""Information boundaries and actor/critic routing for local reservoir policies."""
import unittest

import numpy as np
import torch

import test_seasonal_operation as seasonal_tests
from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class LocalObservationTests(unittest.TestCase):
    def fixture(self, seasonal=True):
        data, config, rules = seasonal_tests.SeasonalExecutionTests().fixture()
        config['seasonal_constraints'] = seasonal
        env = CascadeReservoirEnv(data, config)
        return data, config, rules, env

    def test_feature_values_and_boundary_placeholders(self):
        data, _, _, env = self.fixture()
        env.storage = data.vmax * np.array([.4, .5, .6, .7, .8])
        env.previous_release = data.release_cap * np.array([.1, .2, .3, .4, .5])
        data.external[0] = [1., 2., 3., 4., 5.]
        obs = env.actor_observations()
        self.assertEqual(obs.shape, (5, 16))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(len(env.observation_features), 16)
        expected = [.5, 2., .2, .4, 1., .1, .6, .4,
                    0., 1., 2./366, 0., 1., 0., 1., 5./366]
        np.testing.assert_allclose(obs[1], expected, atol=1e-7, rtol=0.)
        np.testing.assert_array_equal(obs[0, 3:6], [0., 0., 0.])
        np.testing.assert_array_equal(obs[4, 6:8], [0., 1.])
        # Downstream room is signed when today's ceiling requires drawdown.
        env.storage_upper[0, 2] = .4 * data.vmax[2]
        self.assertAlmostEqual(float(env.actor_observations()[1, 7]), -.2, places=6)

    def test_non_neighbor_changes_cannot_reach_an_actor(self):
        for seasonal in (False, True):
            for actor in range(5):
                for remote in range(5):
                    if abs(actor-remote) <= 1:
                        continue
                    with self.subTest(seasonal=seasonal, actor=actor, remote=remote):
                        data, _, _, env = self.fixture(seasonal)
                        before, state_before = env.actor_observations(), env.state()
                        env.storage[remote] *= .5
                        env.previous_release[remote] = 1.
                        data.external[0, remote] += 3.
                        if seasonal:
                            env.storage_upper[:, remote] *= .8
                        np.testing.assert_array_equal(env.actor_observations()[actor], before[actor])
                        self.assertFalse(np.array_equal(env.state(), state_before))

    def test_neighbors_are_selected_relative_to_the_actor(self):
        data, _, _, env = self.fixture()
        before = env.actor_observations()
        env.storage[2] *= .5
        env.previous_release[2] = data.release_cap[2] * .25
        data.external[0, 2] += 2.
        after = env.actor_observations()
        # Reservoir 2 is downstream of actor 1 and upstream of actor 3.
        self.assertNotEqual(float(after[1, 6]), float(before[1, 6]))
        np.testing.assert_allclose(after[3, 3:6], [.5, 2., .25], atol=1e-7)
        np.testing.assert_array_equal(after[0], before[0])
        np.testing.assert_array_equal(after[4], before[4])

    def test_future_inflows_and_soft_targets_do_not_enter_observations(self):
        data, _, rules, env = self.fixture()
        before = env.actor_observations()
        data.external[1:] += 1000.
        for key, values in rules.reference.items():
            values[:] = 0.
        np.testing.assert_array_equal(env.actor_observations(), before)

    def test_reset_and_terminal_keep_actor_and_critic_spaces_distinct(self):
        for seasonal in (False, True):
            with self.subTest(seasonal=seasonal):
                _, _, _, env = self.fixture(seasonal)
                initial, state, _ = env.reset()
                actor_size, state_size = (16, 43) if seasonal else (11, 18)
                self.assertEqual(initial.shape, (5, actor_size))
                self.assertEqual(state.shape, (5, state_size))
                self.assertEqual(env.observation_space[0].shape, (actor_size,))
                self.assertEqual(env.share_observation_space[0].shape, (state_size,))
                for _ in range(5):
                    obs, state, _, done, _, _ = env.step(np.zeros(5))
                self.assertTrue(done.all())
                np.testing.assert_array_equal(obs, np.zeros((5, actor_size)))
                np.testing.assert_array_equal(state, np.zeros((5, state_size)))
                reset, _, _ = env.reset()
                np.testing.assert_array_equal(reset, initial)

    def test_training_buffers_receive_each_actors_own_local_input(self):
        data, physical, _, env = self.fixture()
        data.training_years = (1965,)
        cfg, _ = load_configuration()
        cfg['model']['hidden_sizes'] = [8, 8]
        cfg['algo']['ppo_epoch'] = cfg['algo']['critic_epoch'] = 1
        runner = CascadeReservoirRunner(data, physical, cfg, 7)
        initial, state = env.actor_observations(), env.state()
        runner.collect_episode(1965)
        for i, buffer in enumerate(runner.actor_buffer):
            np.testing.assert_array_equal(buffer.obs[0, 0], initial[i])
            self.assertEqual(buffer.obs.shape, (6, 1, 16))
        np.testing.assert_array_equal(runner.critic_buffer.share_obs[0, 0], state)
        self.assertEqual(runner.critic_buffer.share_obs.shape, (6, 1, 43))
        runner.prep_training()
        actors, critic = runner.train()
        for row in actors+[critic]:
            self.assertTrue(all(np.isfinite(float(value)) for value in row.values()))

    def test_frozen_evaluation_uses_the_same_local_inputs_as_rollout(self):
        data, physical, _, env = self.fixture()
        data.validation_years, data.test_years = (1965,), ()
        expected = []
        while env.index < env.stop:
            expected.append(env.actor_observations())
            env.step(np.zeros(5))
        case = self

        class CaptureActor:
            args = {'action_bins': 200}

            def __init__(self, agent):
                self.agent, self.day = agent, 0

            def prep_rollout(self):
                pass

            def act(self, obs, deterministic):
                case.assertTrue(deterministic)
                case.assertEqual(obs.shape, (1, 16))
                np.testing.assert_array_equal(obs[0], expected[self.day][self.agent])
                self.day += 1
                return torch.zeros((1, 1), dtype=torch.int64)

        actors = [CaptureActor(i) for i in range(5)]
        row = evaluate_policy(actors, data, physical, 1965, 1, 0, 0, gamma=.99)
        self.assertEqual([actor.day for actor in actors], [5]*5)
        self.assertEqual(row['actor_observation_layout'], 'local_neighbors_v1')
        self.assertEqual(row['actor_observation_size'], 16)
        self.assertEqual(row['critic_state_size'], 43)


if __name__ == '__main__':
    unittest.main()
