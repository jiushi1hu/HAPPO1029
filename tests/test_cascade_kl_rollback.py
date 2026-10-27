"""Candidate KL rejection must restore learning state and HAPPO correction factors."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.train_cascade import load_configuration, main
import examples.compare_cascade_policy_budget as comparison
import harl.algorithms.actors.happo as happo_module
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_cascade_reward_penalties as reward_fixtures


class KLRollbackTests(unittest.TestCase):
    def fixture(self, *, rollback=True, target=1e-12, epochs=5, batches=1, lr=.02, gamma=.99):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': [8, 8],
                'ppo_epoch': epochs, 'target_kl': target, 'kl_rollback': rollback,
                'actor_num_mini_batch': batches, 'lr': lr, 'gamma': gamma}
        obs = Box(-1., 1., (3,), dtype=np.float32)
        actor = happo_module.HAPPO(args, obs, Discrete(3))
        buffer = OnPolicyActorBuffer(dict(episode_length=24, n_rollout_threads=1), obs, actor.act_space)
        buffer.obs[:] = np.random.RandomState(5).normal(size=buffer.obs.shape)
        buffer.actions[:, 0, 0] = np.arange(24) % 3
        buffer.update_factor(np.ones_like(buffer.actions))
        self.refresh_likelihoods(actor, buffer)
        advantage = np.where(buffer.actions == 0., 1., -.5).astype(np.float32)
        return actor, buffer, advantage

    @staticmethod
    def refresh_likelihoods(actor, buffer):
        with torch.no_grad():
            lp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
        buffer.action_log_probs[:, 0] = lp.numpy()

    @staticmethod
    def snapshot(actor):
        return copy.deepcopy((actor.actor.state_dict(), actor.actor_optimizer.state_dict(),
                              [None if p.grad is None else p.grad.clone() for p in actor.actor.parameters()]))

    def assert_state_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_validation_requires_a_real_boolean_and_positive_kl_target(self):
        validate = happo_module.validate_kl_rollback
        self.assertFalse(validate(False, None))
        self.assertTrue(validate(True, .01))
        for enabled in (0, 1, 'true', None, np.bool_(True)):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                validate(enabled, .01)
        for target in (None, 0., -1., np.nan, np.inf, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate(True, target)

    def test_real_first_candidate_rejection_restores_fresh_and_populated_adam_and_gradients(self):
        for warm in (False, True):
            with self.subTest(existing_adam_momentum=warm):
                actor, buffer, advantage = self.fixture()
                if warm:
                    sample = next(buffer.feed_forward_generator_actor(
                        advantage, 1, objective_weights=buffer.discount_weights(.99)))
                    actor.update(sample)
                    self.refresh_likelihoods(actor, buffer)
                original = self.snapshot(actor)
                old_lp = buffer.action_log_probs.copy()
                reference = actor.rollout_distribution(buffer)
                info = actor.train(buffer, advantage, 'EP', diagnostics=True)
                self.assert_state_equal(original, self.snapshot(actor))
                np.testing.assert_array_equal(old_lp, buffer.action_log_probs)
                self.assertEqual(info['optimizer_steps'], 0.)
                self.assertEqual(info['optimizer_attempts'], 1.)
                self.assertEqual(info['optimizer_rejected_steps'], 1.)
                self.assertEqual(info['ppo_epochs_completed'], 0.)
                self.assertEqual(info['ppo_epochs_attempted'], 1.)
                self.assertEqual(info['kl_early_stopped'], 1.)
                self.assertGreater(info['rejected_policy_kl'], actor.target_kl)
                self.assertLessEqual(actor.rollout_kl(buffer, reference), actor.target_kl)
                self.assertLessEqual(info['policy_kl'], actor.target_kl)
                self.assertEqual(info['clipped_surrogate_delta'], 0.)
                for key in ('policy_loss', 'dist_entropy', 'actor_grad_norm', 'ratio'):
                    self.assertEqual(float(info[key]), 0.)

    def test_partial_acceptance_keeps_last_accepted_state_and_averages_only_accepted_steps(self):
        actor, buffer, advantage = self.fixture(target=.01, epochs=5, lr=.0003)
        initial_distribution = actor.rollout_distribution(buffer)
        real_update, updates, references = actor.update, [], []

        def update(sample):
            result = real_update(sample)
            updates.append((self.snapshot(actor), [float(v) for v in result[:3]], float(result[3].mean())))
            return result

        def kl(rollout, reference):
            self.assertIs(rollout, buffer)
            torch.testing.assert_close(reference.logits, initial_distribution.logits, rtol=0, atol=0)
            references.append(reference)
            return .02 if len(references) == 2 else .001

        with patch.object(actor, 'update', side_effect=update), patch.object(actor, 'rollout_kl', side_effect=kl):
            info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(len(updates), 2)
        self.assertEqual(len({id(ref) for ref in references}), 1)
        self.assert_state_equal(updates[0][0], self.snapshot(actor))
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['optimizer_attempts'], 2.)
        self.assertEqual(info['optimizer_rejected_steps'], 1.)
        self.assertEqual(info['ppo_epochs_completed'], 1.)
        self.assertEqual(info['ppo_epochs_attempted'], 2.)
        self.assertEqual(info['policy_kl'], .001)
        for key, value in zip(('policy_loss', 'dist_entropy', 'actor_grad_norm'), updates[0][1]):
            self.assertEqual(float(info[key]), value)
        self.assertEqual(float(info['ratio']), updates[0][2])

    def test_multiminibatch_guard_checks_each_step_and_exits_inner_and_outer_loops(self):
        actor, buffer, advantage = self.fixture(target=1e-12, epochs=5, batches=3)
        saved = self.snapshot(actor)
        with patch.object(actor, 'update', wraps=actor.update) as update:
            info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(update.call_count, 1)
        self.assertEqual(info['optimizer_attempts'], 1.)
        self.assertEqual(info['optimizer_steps'], 0.)
        self.assertEqual(info['ppo_epochs_completed'], 0.)
        self.assert_state_equal(saved, self.snapshot(actor))

    def test_partial_minibatch_epoch_keeps_accepted_step_without_counting_a_complete_epoch(self):
        actor, buffer, advantage = self.fixture(target=.01, epochs=5, batches=3, lr=.0003)
        original, updates, calls = actor.update, [], []

        def update(sample):
            result = original(sample)
            updates.append(self.snapshot(actor))
            return result

        def kl(rollout, reference):
            calls.append(reference)
            return .02 if len(calls) == 2 else .001

        with patch.object(actor, 'update', side_effect=update), patch.object(actor, 'rollout_kl', side_effect=kl):
            info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(len(updates), 2)
        self.assert_state_equal(updates[0], self.snapshot(actor))
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['optimizer_attempts'], 2.)
        self.assertEqual(info['optimizer_rejected_steps'], 1.)
        self.assertEqual(info['ppo_epochs_completed'], 0.)
        self.assertEqual(info['ppo_epochs_attempted'], 1.)

    def test_nonbinding_guard_preserves_legacy_weights_moments_gradients_and_rng(self):
        actor, buffer, advantage = self.fixture(target=100., epochs=3, batches=2, lr=.0003)
        reference = copy.deepcopy(actor)
        reference.kl_rollback = False
        outcomes, infos = [], []
        for current in (actor, reference):
            torch.manual_seed(18)
            infos.append(current.train(buffer, advantage, 'EP'))
            outcomes.append((self.snapshot(current), torch.get_rng_state().clone()))
        self.assert_state_equal(outcomes[0], outcomes[1])
        for key in ('policy_loss', 'dist_entropy', 'actor_grad_norm', 'ratio', 'optimizer_steps', 'ppo_epochs_completed'):
            self.assertEqual(float(infos[0][key]), float(infos[1][key]))
        self.assertEqual(infos[0]['optimizer_attempts'], 6.)
        self.assertEqual(infos[0]['optimizer_rejected_steps'], 0.)

    def test_legacy_early_stop_still_retains_the_crossing_update(self):
        actor, buffer, advantage = self.fixture(rollback=False)
        initial = copy.deepcopy(actor.actor.state_dict())
        info = actor.train(buffer, advantage, 'EP')
        self.assertEqual(info['optimizer_steps'], 1.)
        self.assertEqual(info['ppo_epochs_completed'], 1.)
        self.assertGreater(info['policy_kl'], actor.target_kl)
        self.assertTrue(any(not torch.equal(value, actor.actor.state_dict()[name]) for name, value in initial.items()))

    def test_forced_and_zero_objective_rows_never_attempt_or_reject_updates(self):
        for forced in (True, False):
            actor, buffer, advantage = self.fixture(gamma=0.)
            sample = next(buffer.feed_forward_generator_actor(
                advantage, 1, objective_weights=np.ones_like(buffer.actions)))
            actor.update(sample)
            self.refresh_likelihoods(actor, buffer)
            buffer.active_masks[0] = 0.
            if forced:
                buffer.active_masks.fill(0.)
            saved = self.snapshot(actor)
            rng = torch.get_rng_state().clone()
            legacy = copy.deepcopy(actor)
            legacy.kl_rollback = False
            legacy.train(buffer, advantage, 'EP')
            legacy_rng = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            with patch.object(actor, '_snapshot_actor_update', side_effect=AssertionError('snapshot of inactive batch')), \
                    patch.object(actor, 'rollout_kl', wraps=actor.rollout_kl) as kl:
                info = actor.train(buffer, advantage, 'EP')
            self.assert_state_equal(saved, self.snapshot(actor))
            torch.testing.assert_close(legacy_rng, torch.get_rng_state(), rtol=0, atol=0)
            if forced:
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            else:
                self.assertEqual(info['ppo_epochs_completed'], actor.ppo_epoch)
                self.assertEqual(info['ppo_epochs_attempted'], actor.ppo_epoch)
            for key in ('optimizer_steps', 'optimizer_attempts', 'optimizer_rejected_steps'):
                self.assertEqual(info[key], 0.)
            kl.assert_not_called()

    def test_kl_exception_rolls_back_before_reraising(self):
        actor, buffer, advantage = self.fixture()
        sample = next(buffer.feed_forward_generator_actor(
            advantage, 1, objective_weights=buffer.discount_weights(.99)))
        actor.update(sample)
        self.refresh_likelihoods(actor, buffer)
        saved = self.snapshot(actor)
        with patch.object(actor, 'rollout_kl', side_effect=FloatingPointError('invalid candidate KL')), \
                self.assertRaisesRegex(FloatingPointError, 'candidate KL'):
            actor.train(buffer, advantage, 'EP')
        self.assert_state_equal(saved, self.snapshot(actor))

    def test_nonfinite_candidate_parameters_roll_back_before_failure(self):
        actor, buffer, advantage = self.fixture()
        saved, original = self.snapshot(actor), actor.update

        def invalid(sample):
            result = original(sample)
            with torch.no_grad():
                next(actor.actor.parameters()).fill_(float('nan'))
            return result

        with patch.object(actor, 'update', side_effect=invalid), self.assertRaises((FloatingPointError, ValueError)):
            actor.train(buffer, advantage, 'EP')
        self.assert_state_equal(saved, self.snapshot(actor))

    def test_update_exception_after_mutating_parameters_and_adam_restores_both(self):
        actor, buffer, advantage = self.fixture()
        saved, original = self.snapshot(actor), actor.update

        def failed(sample):
            original(sample)
            raise RuntimeError('candidate optimizer failure')

        with patch.object(actor, 'update', side_effect=failed), \
                self.assertRaisesRegex(RuntimeError, 'candidate optimizer failure'):
            actor.train(buffer, advantage, 'EP')
        self.assert_state_equal(saved, self.snapshot(actor))

    def test_sequential_happo_factors_use_restored_actor_weights(self):
        cfg, _ = load_configuration()
        cfg['model']['hidden_sizes'] = [8, 8]
        cfg['algo'].update(ppo_epoch=3, critic_epoch=1, target_kl=.01,
                           kl_rollback=True, fixed_order=True)
        cfg['experiment']['rollout_episodes'] = 2
        data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
        runner = CascadeReservoirRunner(data, physical, cfg, 3)
        with redirect_stdout(io.StringIO()):
            runner.collect_batch(2)
        initial = [copy.deepcopy(a.actor.state_dict()) for a in runner.actor]
        old_logp = [b.action_log_probs.copy() for b in runner.actor_buffer]
        old_likelihoods = []
        for actor, buffer in zip(runner.actor, runner.actor_buffer):
            with torch.no_grad():
                lp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
            old_likelihoods.append(lp.numpy().copy())
        accepted = {}
        with ExitStack() as stack:
            for i, actor in enumerate(runner.actor):
                calls = []
                original = actor.update
                def update(sample, current=actor, operation=original, agent=i):
                    result = operation(sample)
                    if agent not in accepted:
                        accepted[agent] = self.snapshot(current)
                    return result
                def kl(rollout, reference, seen=calls):
                    seen.append(reference)
                    return .02 if len(seen) == 2 else .001
                stack.enter_context(patch.object(actor, 'update', side_effect=update))
                stack.enter_context(patch.object(actor, 'rollout_kl', side_effect=kl))
            runner.prep_training()
            infos, _ = runner.train()
        self.assertTrue(any(info['optimizer_rejected_steps'] for info in infos))
        factor = np.ones((runner.length, 1, 1), dtype=np.float32)
        for i, (actor, buffer) in enumerate(zip(runner.actor, runner.actor_buffer)):
            self.assert_state_equal(accepted[i], self.snapshot(actor))
            np.testing.assert_allclose(buffer.factor, factor, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(buffer.action_log_probs, old_logp[i])
            with torch.no_grad():
                lp, _, _ = actor.evaluate_actions(buffer.obs[:-1, 0], buffer.actions[:, 0])
            factor *= np.where(buffer.active_masks[:-1] > 0,
                               np.exp(lp.numpy()-old_likelihoods[i])[:, None], 1.)
        self.assertTrue(any(not torch.equal(value, actor.actor.state_dict()[name])
                            for actor, weights in zip(runner.actor, initial) for name, value in weights.items()))

    def test_cli_rejects_rollback_without_a_kl_budget_before_data_work(self):
        for options in (['--kl-rollback'], ['--kl-rollback', '--no-kl-limit']):
            with self.subTest(options=options), \
                    patch('sys.argv', ['train_cascade.py', '--check-data-only']+options), \
                    patch('examples.train_cascade.CascadeData') as data, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                main()
            self.assertEqual(failure.exception.code, 2)
            data.assert_not_called()

    def test_cli_enables_and_disables_rollback_before_validation(self):
        for enabled, options in ((True, ['--target-kl', '.01', '--kl-rollback']),
                                 (False, ['--no-kl-rollback'])):
            cfg, _ = load_configuration()
            cfg['algo']['kl_rollback'] = not enabled
            data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
            with patch('sys.argv', ['train_cascade.py', '--check-data-only']+options), \
                    patch('examples.train_cascade.load_configuration', return_value=(cfg, physical)), \
                    patch('examples.train_cascade.CascadeData', return_value=data), \
                    patch('examples.train_cascade.validate_operating_feasibility', return_value=[1965]), \
                    redirect_stdout(io.StringIO()):
                main()
            self.assertIs(cfg['algo']['kl_rollback'], enabled)

    def test_comparison_cli_persists_accepted_and_rejected_step_metrics(self):
        cfg, _ = load_configuration()
        cfg['model']['hidden_sizes'] = [8, 8]
        cfg['algo']['critic_epoch'] = 1
        data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
        data.hashes = {'fixture': 'short-five-day-cascade'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rollback.jsonl'
            argv = ['compare_cascade_policy_budget.py', '--kl-only', '--candidate-kl-rollback',
                    '--candidate-target-kl', '1e-12', '--year', '1965', '--updates', '1',
                    '--rollout-episodes', '1', '--seeds', '1', '--evaluation-repeats', '1',
                    '--output', str(path)]
            with patch('sys.argv', argv), \
                    patch.object(comparison, 'load_configuration', return_value=(cfg, physical)), \
                    patch.object(comparison, 'CascadeData', return_value=data), \
                    patch.object(comparison, 'validate_operating_feasibility', return_value=[1965]), \
                    redirect_stdout(io.StringIO()):
                comparison.main()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        metadata = rows[0]
        self.assertEqual(metadata['comparison_type'], 'kl_rollback')
        self.assertEqual(metadata['kl_rollback_by_arm'], dict(baseline=False, guarded=True))
        arms = {r['arm']: r for r in rows if r['record_type'] == 'paired_check_result'}
        self.assertGreater(sum(arms['guarded']['actor_optimizer_rejected_steps']), 0.)
        self.assertEqual(sum(arms['baseline']['actor_optimizer_rejected_steps']), 0.)
        reports = [r for r in rows if r['record_type'] == 'training_diagnostics' and r['arm'] == 'guarded']
        self.assertEqual(len(reports), 1)
        for i, actor in enumerate(reports[0]['actors']):
            info = actor['optimization']
            self.assertEqual(info['optimizer_attempts'],
                             info['optimizer_steps']+info['optimizer_rejected_steps'])
            self.assertLessEqual(info['policy_kl'], 1e-12)
            for record_key, info_key in (
                    ('actor_optimizer_attempts', 'optimizer_attempts'),
                    ('actor_optimizer_steps', 'optimizer_steps'),
                    ('actor_optimizer_rejected_steps', 'optimizer_rejected_steps')):
                self.assertEqual(arms['guarded'][record_key][i], info[info_key])


if __name__ == '__main__':
    unittest.main()
