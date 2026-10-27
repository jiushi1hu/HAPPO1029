"""Correctness fixes must preserve rollout rows and selected-checkpoint identity."""
import copy
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.train_cascade import main
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner, configure_calendar
import test_reward_scale_metrics as fixtures


class ActorMinibatchSafetyTests(unittest.TestCase):
    def buffer(self):
        buffer = OnPolicyActorBuffer(dict(episode_length=4, n_rollout_threads=2),
                                     Box(-1., 1., (1,), dtype=np.float32), Discrete(3))
        buffer.obs[:-1] = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
        buffer.update_factor(np.ones_like(buffer.actions))
        return buffer

    def test_invalid_or_nondividing_batches_fail_before_shuffle(self):
        buffer = self.buffer()
        for count in (0, -1, True, np.bool_(False), 1.5, '2', 3, 9):
            state = torch.get_rng_state().clone()
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'minibatch count'):
                list(buffer.feed_forward_generator_actor(np.ones_like(buffer.actions), count,
                                                         buffer.discount_weights(.99)))
            torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)

    def test_valid_batches_keep_every_row_and_legacy_shuffle_rng(self):
        buffer = self.buffer()
        torch.manual_seed(35)
        initial = torch.get_rng_state().clone()
        expected_order = torch.randperm(8).numpy()
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial)
        samples = list(buffer.feed_forward_generator_actor(np.ones_like(buffer.actions),
                        np.int64(4), buffer.discount_weights(.99)))
        observed = np.concatenate([sample[0][:, 0] for sample in samples])
        np.testing.assert_array_equal(observed, expected_order)
        np.testing.assert_array_equal(np.sort(observed), np.arange(8))
        self.assertEqual([len(sample[0]) for sample in samples], [2]*4)
        torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_unrestricted_actions_can_omit_availability_masks(self):
        buffer = self.buffer()
        buffer.available_actions = None
        samples = list(buffer.feed_forward_generator_actor(np.ones_like(buffer.actions),
                        2, buffer.discount_weights(.99)))
        self.assertTrue(all(sample[6] is None for sample in samples))
        self.assertEqual(sum(len(sample[0]) for sample in samples), 8)


class CheckpointAttributionTests(unittest.TestCase):
    def fixture(self):
        data, physical, config = fixtures.RewardScaleMetricsTests().fixture()
        data.validation_years = (1965,)
        data.test_years, data.test_labels = (2006,), ('dry',)
        config['experiment']['diagnostics'] = False
        return data, physical, config

    def runner(self):
        data, physical, config = self.fixture()
        return CascadeReservoirRunner(data, physical, config, 4)

    @staticmethod
    def evaluator(actors, data, physical, year, seed, episode, steps, **kwargs):
        return dict(year=year, episode=episode, training_steps=steps, seed=seed,
                    energy_MWh=100e6, energy_TWh=100., episode_reward=10. if episode == 294 else 1.,
                    episode_penalty=0.)

    def test_final_test_uses_selected_episode_steps_and_update_with_current_run_cost_separate(self):
        runner = self.runner()
        runner.iteration, runner.update_count, runner.training_steps = 294, 49, 107374
        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=self.evaluator), \
                redirect_stdout(io.StringIO()):
            runner.validate()
            selected = [copy.deepcopy(a.actor.state_dict()) for a in runner.actor]
            with torch.no_grad():
                for actor in runner.actor:
                    next(actor.actor.parameters()).add_(.1)
            runner.iteration, runner.update_count, runner.training_steps = 300, 50, 109567
            runner.validate()
            result = runner.final_test()[0]
        self.assertEqual(result['episode'], 294)
        self.assertEqual(result['training_steps'], 107374)
        self.assertEqual(result['selected_update'], 49)
        self.assertEqual(result['run_training_steps'], 109567)
        self.assertEqual(result['typical_label'], 'dry')
        for actor, state in zip(runner.actor, selected):
            for key, value in state.items():
                torch.testing.assert_close(value, actor.actor.state_dict()[key], rtol=0, atol=0)

    def test_untrained_checkpoint_has_zero_selected_steps_without_fallback_to_final_budget(self):
        runner = self.runner()
        def evaluated(*args, **kwargs):
            row = self.evaluator(*args, **kwargs)
            row['episode_reward'] = 10. if args[6] == 0 else 1.
            return row
        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', side_effect=evaluated), \
                redirect_stdout(io.StringIO()):
            runner.validate()
            runner.iteration, runner.update_count, runner.training_steps = 6, 1, 2191
            runner.validate()
            row = runner.final_test()[0]
        self.assertEqual((row['episode'], row['training_steps'], row['selected_update']), (0, 0, 0))
        self.assertEqual(row['run_training_steps'], 2191)

    def test_fixed_year_selection_captures_matching_training_metadata(self):
        data, physical, config = self.fixture()
        config['experiment']['fixed_training_year'] = 1965
        runner = CascadeReservoirRunner(data, physical, config, 4)
        runner.iteration, runner.update_count, runner.training_steps = 12, 2, 4380
        row = dict(year=1965, energy_MWh=100e6, energy_TWh=100., episode_reward=3.,
                   episode_generation_reward=4., episode_penalty_guaranteed_output=.5,
                   episode_penalty_ecological_flow=.5, feasible=True, violation_count=0)
        with patch('harl.runners.cascade_reservoir_runner.evaluate_policy', return_value=row), \
                redirect_stdout(io.StringIO()):
            runner.evaluate_training_year()
        self.assertEqual((runner.best_iteration, runner.best_training_steps, runner.best_update), (12, 4380, 2))

    def test_unsupported_critic_combination_is_rejected_before_data_or_model_work(self):
        data, physical, config = self.fixture()
        config['algo']['use_clipped_value_loss'] = True
        config['train']['use_valuenorm'] = True
        with patch('sys.argv', ['train_cascade.py', '--check-data-only']), \
                patch('examples.train_cascade.load_configuration', return_value=(config, physical)), \
                patch('examples.train_cascade.CascadeData') as loader, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)
        loader.assert_not_called()
        with patch('harl.runners.cascade_reservoir_runner.CascadeReservoirEnv') as env, \
                patch('harl.runners.cascade_reservoir_runner.torch.manual_seed') as seed, \
                self.assertRaisesRegex(ValueError, 'Clipped value loss with ValueNorm'):
            CascadeReservoirRunner(data, physical, config, 4)
        env.assert_not_called()
        seed.assert_not_called()

    def test_calendar_rejects_invalid_minibatch_types_and_preserves_integer_divisor_adjustment(self):
        data, physical, config = self.fixture()
        for key in ('actor_num_mini_batch', 'critic_num_mini_batch'):
            for value in (True, np.bool_(True), 0, -1, 1.5, '2', None):
                invalid = copy.deepcopy(config)
                invalid['algo'][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'positive integers'):
                    configure_calendar(invalid, data)
            config['algo'][key] = np.int64(3)
        adjusted = configure_calendar(config, data)
        length = adjusted['train']['episode_length']
        expected = max(n for n in range(1, min(3, length)+1) if length % n == 0)
        for key in ('actor_num_mini_batch', 'critic_num_mini_batch'):
            self.assertEqual(adjusted['algo'][key], expected)
            self.assertEqual(config['algo'][key], 3)


if __name__ == '__main__':
    unittest.main()
