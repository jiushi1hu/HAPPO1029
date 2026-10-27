"""The shared KL guard measures and retains one common-rollout joint policy."""
import copy
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box, Discrete

from examples.train_cascade import load_configuration, main
from harl.algorithms.actors.happo import HAPPO, validate_joint_target_kl
from harl.algorithms.actors.joint_kl import JointKLBudget
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
import test_cascade_reward_penalties as reward_fixtures


class JointKLTests(unittest.TestCase):
    def fixture(self, *, joint=100., epochs=2, batches=1, lr=.02, warm=False):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo'], 'hidden_sizes': [8, 8],
                'ppo_epoch': epochs, 'target_kl': 100., 'kl_rollback': True,
                'joint_target_kl': joint, 'actor_num_mini_batch': batches, 'lr': lr}
        obs = Box(-1., 1., (3,), dtype=np.float32)
        actor = HAPPO(args, obs, Discrete(3))
        buffer = OnPolicyActorBuffer(dict(episode_length=24, n_rollout_threads=1), obs, actor.act_space)
        buffer.obs[:] = np.random.RandomState(5).normal(size=buffer.obs.shape)
        buffer.actions[:, 0, 0] = np.arange(24) % 3
        buffer.update_factor(np.ones_like(buffer.actions))
        advantage = np.where(buffer.actions == 0., 1., -.5).astype(np.float32)
        if warm:
            actor.update(next(buffer.feed_forward_generator_actor(
                advantage, 1, objective_weights=buffer.discount_weights(.99))))
        self.refresh_likelihoods(actor, buffer)
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

    def test_joint_target_validation_requires_positive_finite_limit_and_rollback(self):
        self.assertIsNone(validate_joint_target_kl(None, False))
        self.assertEqual(validate_joint_target_kl('.03', True), .03)
        for target in (0., -1., np.nan, np.inf, True, np.bool_(True), 'invalid', 10**400):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_joint_target_kl(target, True)
        with self.assertRaisesRegex(ValueError, 'kl_rollback'):
            validate_joint_target_kl(.01, False)

    def test_additive_common_row_kl_equals_explicit_product_policy_kl(self):
        actor, buffer, _ = self.fixture()
        actors = [actor, copy.deepcopy(actor)]
        buffers = [buffer, copy.deepcopy(buffer)]
        budget = JointKLBudget(actors, buffers, 100.)
        frozen = [reference.logits.clone() for reference in budget.before]
        with torch.no_grad():
            actors[0].actor.act.action_out.linear.bias.add_(torch.tensor([.6, -.4, .2]))
            actors[1].actor.act.action_out.linear.bias.add_(torch.tensor([-.3, .7, -.5]))
        current = [a.rollout_distribution(b) for a, b in zip(actors, buffers)]
        old_product = (budget.before[0].probs[:, :, None]*budget.before[1].probs[:, None, :]).flatten(1)
        new_product = (current[0].probs[:, :, None]*current[1].probs[:, None, :]).flatten(1)
        explicit = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=old_product),
            torch.distributions.Categorical(probs=new_product)).mean()
        components = [budget.for_actor(i).actor_kl() for i in range(2)]
        self.assertAlmostEqual(sum(components), float(explicit), places=6)
        for i, component in enumerate(components):
            budget.for_actor(i).commit(component)
            torch.testing.assert_close(frozen[i], budget.before[i].logits, rtol=0, atol=0)
        self.assertAlmostEqual(budget.total_kl, float(explicit), places=6)

    def test_actor_specific_controllability_means_are_not_used_as_joint_components(self):
        actor, buffer, _ = self.fixture()
        actors = [actor, copy.deepcopy(actor)]
        buffers = [buffer, copy.deepcopy(buffer)]
        budget = JointKLBudget(actors, buffers, 100.)
        with torch.no_grad():
            for a in actors:
                a.actor.act.action_out.linear.weight[0].add_(torch.arange(8)*.17)
        raw, masked = [], []
        for i, (a, b) in enumerate(zip(actors, buffers)):
            divergence = torch.distributions.kl_divergence(budget.before[i], a.rollout_distribution(b))
            active = np.argsort(divergence.numpy())[-6:]
            b.active_masks.fill(0.)
            b.active_masks[active, 0, 0] = 1.
            raw.append(budget.for_actor(i).actor_kl())
            masked.append(a.rollout_kl(b, budget.before[i]))
        self.assertGreater(sum(masked)-sum(raw), .001)
        self.assertAlmostEqual(raw[0], raw[1], places=7)

    def test_constructor_rejects_misaligned_rows_and_shared_parameters(self):
        actor, buffer, _ = self.fixture()
        second = copy.deepcopy(actor)
        bad = copy.deepcopy(buffer)
        bad.actions = bad.actions[:-1]
        with self.assertRaisesRegex(ValueError, 'aligned'):
            JointKLBudget([actor, second], [buffer, bad], .01)
        with self.assertRaisesRegex(ValueError, 'independent'):
            JointKLBudget([actor, actor], [buffer, copy.deepcopy(buffer)], .01)
        second_buffer = copy.deepcopy(buffer)
        second_buffer.masks[2] = 0.
        with self.assertRaisesRegex(ValueError, 'aligned'):
            JointKLBudget([actor, second], [buffer, second_buffer], .01)

    def test_joint_guard_must_match_the_actor_buffer_and_configured_limit(self):
        actor, buffer, advantage = self.fixture(joint=.01)
        with self.assertRaisesRegex(ValueError, 'joint_kl_guard'):
            actor.train(buffer, advantage, 'EP')
        budget = JointKLBudget([actor], [buffer], .02)
        with self.assertRaisesRegex(ValueError, 'match'):
            actor.train(buffer, advantage, 'EP', joint_kl_guard=budget.for_actor(0))
        budget = JointKLBudget([actor], [buffer], .01)
        with self.assertRaisesRegex(ValueError, 'belong'):
            actor.train(copy.deepcopy(buffer), advantage, 'EP', joint_kl_guard=budget.for_actor(0))

    def test_real_joint_crossing_restores_parameters_adam_gradients_and_budget(self):
        for warm in (False, True):
            with self.subTest(populated_adam=warm):
                actor, buffer, advantage = self.fixture(joint=1e-12, warm=warm)
                budget = JointKLBudget([actor], [buffer], actor.joint_target_kl)
                saved = self.snapshot(actor)
                info = actor.train(buffer, advantage, 'EP', diagnostics=True,
                                   joint_kl_guard=budget.for_actor(0))
                self.assert_state_equal(saved, self.snapshot(actor))
                self.assertEqual(info['optimizer_steps'], 0.)
                self.assertEqual(info['optimizer_attempts'], 1.)
                self.assertEqual(info['optimizer_rejected_steps'], 1.)
                self.assertEqual(info['joint_kl_early_stopped'], 1.)
                self.assertGreater(info['rejected_joint_policy_kl'], actor.joint_target_kl)
                self.assertLess(info['rejected_policy_kl'], actor.target_kl)
                self.assertEqual(info['joint_policy_kl'], 0.)
                self.assertEqual(info['joint_actor_kl'], 0.)
                self.assertEqual(info['clipped_surrogate_delta'], 0.)
                self.assertEqual(budget.total_kl, 0.)

    def test_earlier_accepted_actor_consumes_the_later_actors_remaining_budget(self):
        actor, buffer, advantage = self.fixture(epochs=1, warm=True)
        probe, probe_buffer = copy.deepcopy(actor), copy.deepcopy(buffer)
        reference = probe.rollout_distribution(probe_buffer)
        probe.joint_target_kl = None
        probe.train(probe_buffer, advantage, 'EP')
        candidate = float(torch.distributions.kl_divergence(
            reference, probe.rollout_distribution(probe_buffer)).mean().clamp_min(0.))
        self.assertGreater(candidate, 1e-5)
        target = candidate*1.5
        actors = [actor, copy.deepcopy(actor)]
        buffers = [buffer, copy.deepcopy(buffer)]
        for a in actors:
            a.joint_target_kl = target
        budget = JointKLBudget(actors, buffers, target)
        untouched_second = self.snapshot(actors[1])
        first = actors[0].train(buffers[0], advantage, 'EP', joint_kl_guard=budget.for_actor(0))
        retained_first = self.snapshot(actors[0])
        second = actors[1].train(buffers[1], advantage, 'EP', joint_kl_guard=budget.for_actor(1))
        self.assertEqual(first['optimizer_steps'], 1.)
        self.assertEqual(second['optimizer_steps'], 0.)
        self.assertEqual(second['optimizer_rejected_steps'], 1.)
        self.assertEqual(second['joint_kl_early_stopped'], 1.)
        self.assertGreater(second['rejected_joint_policy_kl'], target)
        self.assertAlmostEqual(second['joint_kl_before'], first['joint_policy_kl'], places=7)
        self.assertAlmostEqual(second['joint_actor_budget'], target-first['joint_policy_kl'], places=7)
        self.assertAlmostEqual(budget.total_kl, first['joint_policy_kl'], places=7)
        self.assertLessEqual(budget.total_kl, target)
        self.assert_state_equal(retained_first, self.snapshot(actors[0]))
        self.assert_state_equal(untouched_second, self.snapshot(actors[1]))

    def test_nonbinding_joint_guard_preserves_learning_state_and_rng_exactly(self):
        actor, buffer, advantage = self.fixture(epochs=3, batches=2, lr=.0003, warm=True)
        legacy = copy.deepcopy(actor)
        legacy.joint_target_kl = None
        old_buffer = copy.deepcopy(buffer)
        rng = torch.get_rng_state().clone()
        legacy_info = legacy.train(old_buffer, advantage, 'EP', diagnostics=True)
        legacy_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        budget = JointKLBudget([actor], [buffer], actor.joint_target_kl)
        info = actor.train(buffer, advantage, 'EP', diagnostics=True, joint_kl_guard=budget.for_actor(0))
        self.assert_state_equal(self.snapshot(legacy), self.snapshot(actor))
        torch.testing.assert_close(legacy_rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(info['optimizer_steps'], 6.)
        self.assertEqual(info['optimizer_rejected_steps'], 0.)
        for key in legacy_info:
            self.assertEqual(float(legacy_info[key]), float(info[key]))
        self.assertAlmostEqual(info['joint_policy_kl'], budget.total_kl, places=9)
        expected = float(torch.distributions.kl_divergence(
            budget.before[0], actor.rollout_distribution(buffer)).mean().clamp_min(0.))
        self.assertEqual(budget.total_kl, expected)

    def test_repeated_candidates_replace_the_current_actors_component(self):
        actor, buffer, advantage = self.fixture(epochs=3, lr=.0003)
        budget = JointKLBudget([actor], [buffer], actor.joint_target_kl)
        guard = budget.for_actor(0)
        operation, measured = guard.actor_kl, []

        def measure():
            value = operation()
            measured.append(value)
            return value

        with patch.object(guard, 'actor_kl', side_effect=measure):
            info = actor.train(buffer, advantage, 'EP', joint_kl_guard=guard)
        self.assertEqual(info['optimizer_steps'], 3.)
        self.assertEqual(len(measured), 3)
        self.assertEqual(budget.total_kl, measured[-1])
        self.assertEqual(info['joint_actor_kl'], measured[-1])
        self.assertGreater(sum(measured), budget.total_kl)

    def test_local_rejection_does_not_commit_an_in_budget_joint_candidate(self):
        actor, buffer, advantage = self.fixture()
        actor.target_kl = 1e-12
        budget = JointKLBudget([actor], [buffer], actor.joint_target_kl)
        saved = self.snapshot(actor)
        info = actor.train(buffer, advantage, 'EP', joint_kl_guard=budget.for_actor(0))
        self.assertEqual(info['optimizer_rejected_steps'], 1.)
        self.assertEqual(info['joint_kl_early_stopped'], 0.)
        self.assertEqual(info['joint_policy_kl'], 0.)
        self.assertGreater(info['rejected_joint_policy_kl'], 0.)
        self.assertEqual(budget.total_kl, 0.)
        self.assert_state_equal(saved, self.snapshot(actor))

    def test_joint_measurement_exception_restores_the_candidate_and_preserves_prior_total(self):
        actor, buffer, advantage = self.fixture(warm=True)
        second, second_buffer = copy.deepcopy(actor), copy.deepcopy(buffer)
        budget = JointKLBudget([actor, second], [buffer, second_buffer], actor.joint_target_kl)
        actor.train(buffer, advantage, 'EP', joint_kl_guard=budget.for_actor(0))
        retained = budget.total_kl
        self.assertGreater(retained, 0.)
        guard, saved, first = budget.for_actor(1), self.snapshot(second), self.snapshot(actor)
        with patch.object(guard, 'actor_kl', side_effect=FloatingPointError('joint measurement failed')), \
                self.assertRaisesRegex(FloatingPointError, 'joint measurement'):
            second.train(second_buffer, advantage, 'EP', joint_kl_guard=guard)
        self.assertEqual(budget.total_kl, retained)
        self.assert_state_equal(saved, self.snapshot(second))
        self.assert_state_equal(first, self.snapshot(actor))

    def test_all_forced_actor_has_no_candidate_and_keeps_existing_joint_total(self):
        actor, buffer, advantage = self.fixture()
        second, second_buffer = copy.deepcopy(actor), copy.deepcopy(buffer)
        second_buffer.active_masks.fill(0.)
        budget = JointKLBudget([actor, second], [buffer, second_buffer], actor.joint_target_kl)
        first = actor.train(buffer, advantage, 'EP', joint_kl_guard=budget.for_actor(0))
        guard, saved = budget.for_actor(1), self.snapshot(second)
        with patch.object(guard, 'actor_kl', side_effect=AssertionError('forced actor candidate')):
            info = second.train(second_buffer, advantage, 'EP', joint_kl_guard=guard)
        self.assertEqual(info['optimizer_attempts'], 0.)
        self.assertEqual(info['joint_actor_kl'], 0.)
        self.assertEqual(info['joint_policy_kl'], first['joint_policy_kl'])
        self.assert_state_equal(saved, self.snapshot(second))

    def test_zero_discount_objective_has_no_joint_candidate_but_preserves_shuffling(self):
        actor, buffer, advantage = self.fixture(warm=True)
        actor.args['gamma'] = 0.
        buffer.active_masks[0] = 0.
        legacy, legacy_buffer = copy.deepcopy(actor), copy.deepcopy(buffer)
        legacy.joint_target_kl = None
        rng = torch.get_rng_state().clone()
        legacy.train(legacy_buffer, advantage, 'EP')
        legacy_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        budget = JointKLBudget([actor], [buffer], actor.joint_target_kl)
        guard, saved = budget.for_actor(0), self.snapshot(actor)
        with patch.object(guard, 'actor_kl', side_effect=AssertionError('zero objective candidate')):
            info = actor.train(buffer, advantage, 'EP', joint_kl_guard=guard)
        self.assertEqual(info['optimizer_attempts'], 0.)
        self.assertEqual(info['joint_policy_kl'], 0.)
        self.assertEqual(info['ppo_epochs_completed'], actor.ppo_epoch)
        self.assert_state_equal(saved, self.snapshot(actor))
        torch.testing.assert_close(legacy_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_sequential_runner_uses_retained_actor_policies_for_joint_kl_and_happo_factors(self):
        cfg, _ = load_configuration()
        cfg['model']['hidden_sizes'] = [8, 8]
        cfg['algo'].update(ppo_epoch=2, critic_epoch=1, target_kl=100.,
                           kl_rollback=True, joint_target_kl=1e-12, fixed_order=True)
        cfg['experiment']['rollout_episodes'] = 2
        data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
        runner = CascadeReservoirRunner(data, physical, cfg, 3)
        with redirect_stdout(io.StringIO()):
            runner.collect_batch(2)
        initial = [self.snapshot(a) for a in runner.actor]
        critic_before = copy.deepcopy(runner.critic.critic.state_dict())
        runner.prep_training()
        infos, _ = runner.train()
        for a, b, state, info in zip(runner.actor, runner.actor_buffer, initial, infos):
            self.assert_state_equal(state, self.snapshot(a))
            np.testing.assert_array_equal(b.factor, np.ones_like(b.factor))
            self.assertEqual(info['joint_policy_kl'], 0.)
            self.assertGreater(info['rejected_joint_policy_kl'], cfg['algo']['joint_target_kl'])
            self.assertEqual(info['optimizer_steps'], 0.)
        self.assertTrue(any(not torch.equal(value, runner.critic.critic.state_dict()[name])
                            for name, value in critic_before.items()))
        self.assertLessEqual(runner.joint_kl_info['policy_kl'], cfg['algo']['joint_target_kl'])

    def test_cli_rejects_joint_budget_without_candidate_rollback_before_data_work(self):
        for options in (['--joint-target-kl', '.02'],
                        ['--joint-target-kl', '.02', '--target-kl', '.01'],
                        ['--joint-target-kl', '.02', '--kl-rollback', '--no-kl-limit'],
                        ['--joint-target-kl', 'nan', '--target-kl', '.01', '--kl-rollback']):
            with self.subTest(options=options), \
                    patch('sys.argv', ['train_cascade.py', '--check-data-only']+options), \
                    patch('examples.train_cascade.CascadeData') as data, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                main()
            self.assertEqual(failure.exception.code, 2)
            data.assert_not_called()

    def test_cli_explicitly_enables_and_disables_joint_budget(self):
        for target, options in ((.02, ['--joint-target-kl', '.02', '--target-kl', '.01', '--kl-rollback']),
                                (None, ['--no-joint-kl-limit', '--no-kl-rollback', '--no-kl-limit'])):
            cfg, _ = load_configuration()
            cfg['algo'].update(joint_target_kl=.03, target_kl=.01, kl_rollback=True)
            data, physical, _ = reward_fixtures.RewardPenaltyTests().fixture()
            with patch('sys.argv', ['train_cascade.py', '--check-data-only']+options), \
                    patch('examples.train_cascade.load_configuration', return_value=(cfg, physical)), \
                    patch('examples.train_cascade.CascadeData', return_value=data), \
                    patch('examples.train_cascade.validate_operating_feasibility', return_value=[1965]), \
                    redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(cfg['algo']['joint_target_kl'], target)


if __name__ == '__main__':
    unittest.main()
