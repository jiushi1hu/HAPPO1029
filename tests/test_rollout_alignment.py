"""Audit real calendar rollouts against independent call-boundary records."""
import unittest
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from gym.spaces import Box

from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm


class RecordingEnv(CascadeReservoirEnv):
    def reset(self):
        self.records = []
        return super().reset()

    def step(self, actions):
        before = self.state().copy()
        actor_before = self.actor_observations().copy()
        date = self.data.dates[self.index]
        result = super().step(actions)
        self.records.append(dict(
            obs=before, actor_obs=actor_before, date=date, sampled=actions.copy(),
            next_obs=self.state().copy(), next_actor_obs=result[0].copy(),
            executed=self.last_transition['actions'].copy(),
            controls=self.last_transition['controls'].copy(),
            reward=result[2].copy(), done=result[3].copy()))
        return result


class RolloutAlignmentTests(unittest.TestCase):
    @staticmethod
    def forward_gae(rewards, values, masks, gamma, trace):
        """Independent finite sum of TD errors, rather than reverse recursion."""
        rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
        delta = rewards + gamma * masks[1:] * values[1:] - values[:-1]
        advantage = np.zeros_like(rewards)
        for t in range(len(rewards)):
            weights = np.concatenate((np.ones_like(masks[:1]),
                np.cumprod(gamma * trace * masks[t + 1:-1], axis=0)), axis=0)
            advantage[t] = (weights * delta[t:]).sum(axis=0)
        return advantage + values[:-1]

    def audit_returns(self, runner):
        cb, normalizer = runner.critic_buffer, runner.value_normalizer
        # Read the affine transform independently of denormalize().
        mean, variance = normalizer.running_mean_var()
        physical_values = cb.value_preds.astype(np.float64) * variance.numpy() ** .5 + mean.numpy()
        expected = self.forward_gae(cb.rewards, physical_values, cb.masks,
                                    cb.gamma, cb.gae_lambda)
        np.testing.assert_allclose(cb.returns[:-1], expected, rtol=2e-6, atol=5e-6)
        return expected

    def test_hand_calculated_gae_and_terminal_bootstrap(self):
        cfg, _ = load_configuration()
        args = {**cfg['train'], **cfg['model'], **cfg['algo'], 'episode_length': 3, 'gamma': 1.}
        self.assertTrue(args['use_gae'])
        self.assertTrue(args['use_proper_time_limits'])
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                norm = ValueNorm(1) if normalized else None
                if norm is not None:
                    norm.update(np.array([[5.], [15.], [25.]], dtype=np.float32))
                cb = OnPolicyCriticBufferEP(args, Box(-np.inf, np.inf, (18,), dtype=np.float32))
                cb.rewards[:, 0, 0] = [1., 2., 3.]
                cb.masks[-1] = 0.
                values = np.array([[10.], [20.], [30.]], dtype=np.float32)
                cb.value_preds[:-1, 0] = norm.normalize(values).numpy() if normalized else values
                # A finite but arbitrarily large terminal prediction must have no effect.
                cb.compute_returns(np.array([[100000.]], dtype=np.float32), norm)
                np.testing.assert_allclose(cb.returns[:-1, 0, 0], [8.0325, 6.35, 3.],
                                           rtol=0, atol=5e-6)
                before = cb.returns.copy()
                cb.compute_returns(np.array([[-100000.]], dtype=np.float32), norm)
                np.testing.assert_array_equal(cb.returns, before)
                # Algebraic endpoint check, not a training-configuration change.
                cb.gae_lambda = 1.
                cb.compute_returns(np.array([[100000.]], dtype=np.float32), norm)
                np.testing.assert_allclose(cb.returns[:-1, 0, 0], [6., 5., 3.],
                                           rtol=0, atol=5e-6)

    def test_gae_cannot_cross_an_episode_boundary(self):
        cfg, _ = load_configuration()
        args = {**cfg['train'], **cfg['model'], **cfg['algo'], 'episode_length': 4, 'gamma': 1.}
        cb = OnPolicyCriticBufferEP(args, Box(-np.inf, np.inf, (18,), dtype=np.float32))
        cb.masks[2] = cb.masks[4] = 0.
        cb.rewards[:, 0, 0] = [1., 2., 3., 4.]
        cb.value_preds[:-1, 0, 0] = [10., 20., 30., 40.]
        cb.compute_returns(np.array([[12345.]], dtype=np.float32))
        np.testing.assert_allclose(cb.returns[:-1, 0, 0], [3.9, 2., 8.8, 4.],
                                   rtol=0, atol=5e-6)
        first_episode = cb.returns[:2].copy()
        cb.rewards[2:] *= 1000.
        cb.value_preds[2:-1] *= -1000.
        cb.compute_returns(np.array([[-12345.]], dtype=np.float32))
        np.testing.assert_array_equal(cb.returns[:2], first_episode)

    def audit_year(self, runner, year):
        actor_calls = [[] for _ in runner.actor]
        critic_calls = []
        normalization_before = copy.deepcopy(runner.value_normalizer.state_dict())

        def record_actor(i, original):
            def call(obs):
                result = original(obs)
                actor_calls[i].append((obs.copy(),
                                       result[0].cpu().numpy().copy(),
                                       result[1].cpu().numpy().copy()))
                return result
            return call

        original_critic = runner.critic.get_values

        def record_critic(obs):
            result = original_critic(obs)
            critic_calls.append((obs.copy(), result.cpu().numpy().copy()))
            return result

        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch('harl.runners.cascade_reservoir_runner.CascadeReservoirEnv', RecordingEnv))
            for i, actor in enumerate(runner.actor):
                stack.enter_context(patch.object(actor, 'get_actions', record_actor(i, actor.get_actions)))
            stack.enter_context(patch.object(runner.critic, 'get_values', record_critic))
            runner.collect_episode(year)

        records = runner.envs[0].records
        cb = runner.critic_buffer
        length = len(records)
        self.assertEqual(length, 366 if year == 1968 else 365)
        self.assertEqual(len(critic_calls), length + 1)  # Includes V(s_T).
        self.assertEqual(records[0]['date'].isoformat(), str(year) + '-01-01')
        self.assertEqual(records[-1]['date'].isoformat(), str(year) + '-12-31')
        self.assertTrue(np.all(cb.masks[:-1] == 1))
        self.assertTrue(np.all(cb.masks[-1] == 0))
        self.assertTrue(np.all(cb.bad_masks == 1))  # Annual objective terminates, not truncates.
        self.assertEqual(cb.step, 0)
        for t, row in enumerate(records):
            np.testing.assert_array_equal(cb.share_obs[t, 0], row['obs'])
            np.testing.assert_array_equal(cb.share_obs[t + 1, 0], row['next_obs'])
            np.testing.assert_array_equal(cb.rewards[t, 0], row['reward'][0])
            np.testing.assert_array_equal(cb.value_preds[t], critic_calls[t][1])
            np.testing.assert_array_equal(critic_calls[t][0], cb.share_obs[t])
            self.assertEqual(bool(row['done'].all()), t == length - 1)
            if t:
                np.testing.assert_array_equal(row['obs'], records[t - 1]['next_obs'])
            for i, b in enumerate(runner.actor_buffer):
                obs, sampled, log_prob = actor_calls[i][t]
                np.testing.assert_array_equal(obs[0], row['actor_obs'][i])
                self.assertEqual(float(b.masks[t+1,0,0]), float(not row['done'].all()))
                np.testing.assert_array_equal(sampled[0], row['sampled'][i])
                np.testing.assert_array_equal(b.obs[t, 0], row['actor_obs'][i])
                np.testing.assert_array_equal(b.obs[t + 1, 0], row['next_actor_obs'][i])
                np.testing.assert_array_equal(b.actions[t, 0], row['executed'][i:i + 1])
                np.testing.assert_array_equal(b.active_masks[t, 0], row['controls'][i:i + 1])
                expected_lp = log_prob if row['controls'][i] else np.zeros_like(log_prob)
                np.testing.assert_array_equal(b.action_log_probs[t], expected_lp)
                if row['controls'][i]:
                    np.testing.assert_array_equal(b.actions[t], sampled)
                else:
                    self.assertEqual(float(b.actions[t, 0, 0]), 0.)

        np.testing.assert_array_equal(cb.value_preds[-1], critic_calls[-1][1])
        np.testing.assert_array_equal(critic_calls[-1][0], cb.share_obs[-1])
        self.assertTrue(np.all(cb.share_obs[-1] == 0))
        np.testing.assert_allclose(cb.returns[-2], cb.rewards[-1], atol=1e-6, rtol=0)
        for key, value in normalization_before.items():
            torch.testing.assert_close(runner.value_normalizer.state_dict()[key], value, rtol=0, atol=0)
        self.audit_returns(runner)
        for i, (actor, b) in enumerate(zip(runner.actor, runner.actor_buffer)):
            self.assertEqual(len(actor_calls[i]), length)
            self.assertEqual(b.step, 0)
            self.assertTrue(np.all(b.masks[:-1] == 1))
            self.assertTrue(np.all(b.masks[-1] == 0))
            self.assertEqual(float(b.active_masks[-2, 0, 0]), 0.)
            # Same frozen policy, same state/index: initial importance ratio is one.
            with torch.no_grad():
                lp, _, _ = actor.evaluate_actions(b.obs[:-1, 0], b.actions[:, 0])
            active = b.active_masks[:-1, 0, 0] > 0
            ratio = np.exp(lp.numpy()[active] - b.action_log_probs[:, 0][active])
            # Batch-size-dependent float32 kernels differ by a few ulps;
            # per-call stored log probabilities were checked exactly above.
            np.testing.assert_allclose(ratio, 1., atol=1e-5, rtol=0)
        self.audit_minibatches(runner)

    def audit_minibatches(self, runner):
        """Identify rows by time-to-go, independently of generator permutations."""
        length, cb = runner.length, runner.critic_buffer

        def indices(obs):
            return length - np.rint(obs[:, -1] * 366).astype(int)

        for b in runner.actor_buffer:
            ids = np.arange(length, dtype=np.float32).reshape(length, 1, 1)
            b.update_factor(ids + 1000)
            for batches in (1, length):  # Full-year batch and one sample per batch.
                seen = []
                for obs, actions, masks, active, lp, adv, available, factor in \
                        b.feed_forward_generator_actor(ids + 2000, batches):
                    ix = indices(obs)
                    seen.extend(ix.tolist())
                    for actual, expected in ((obs, b.obs[ix, 0]),
                            (actions, b.actions[ix, 0]), (masks, b.masks[ix, 0]),
                            (active, b.active_masks[ix, 0]), (lp, b.action_log_probs[ix, 0])):
                        np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(adv[:, 0], ix + 2000)
                    np.testing.assert_array_equal(factor[:, 0], ix + 1000)
                    if b.available_actions is None:
                        self.assertIsNone(available)
                    else:
                        np.testing.assert_array_equal(available,b.available_actions[ix,0])
                self.assertEqual(sorted(seen), list(range(length)))
        for batches in (1, length):
            seen = []
            for obs, values, returns, masks in cb.feed_forward_generator_critic(batches):
                ix = indices(obs)
                seen.extend(ix.tolist())
                for actual, expected in ((obs, cb.share_obs[ix, 0]),
                        (values, cb.value_preds[ix, 0]),
                        (returns, cb.returns[ix, 0]), (masks, cb.masks[ix, 0])):
                    np.testing.assert_array_equal(actual, expected)
            self.assertEqual(sorted(seen), list(range(length)))

    def test_real_rollouts_and_reset_across_update(self):
        cfg, physical = load_configuration()
        data = CascadeData(Path(__file__).resolve().parents[1] / 'data/cascade_reservoir_new',
                           physical, cfg['experiment'])
        runner = CascadeReservoirRunner(data, physical, cfg, 1)
        self.audit_year(runner, 1965)
        previous_buffer = runner.critic_buffer
        runner.prep_training()
        cb = runner.critic_buffer
        returns_before = cb.returns.copy()
        raw_advantages = cb.returns[:-1] - runner.value_normalizer.denormalize(cb.value_preds[:-1])
        expected_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-5)
        norm_before = copy.deepcopy(runner.value_normalizer.state_dict())
        delivered = []

        def check_normalizer():
            for key, value in norm_before.items():
                torch.testing.assert_close(runner.value_normalizer.state_dict()[key], value, rtol=0, atol=0)

        def checked_actor(i, original):
            def call(buffer, advantages, state_type):
                check_normalizer()
                np.testing.assert_array_equal(advantages, expected_advantages)
                delivered.append(i)
                result = original(buffer, advantages, state_type)
                check_normalizer()
                return result
            return call

        original_critic_train = runner.critic.train

        def checked_critic(*args, **kwargs):
            self.assertEqual(sorted(delivered), list(range(5)))
            check_normalizer()
            return original_critic_train(*args, **kwargs)

        from contextlib import ExitStack
        with ExitStack() as stack:
            for i, actor in enumerate(runner.actor):
                stack.enter_context(patch.object(actor, 'train', checked_actor(i, actor.train)))
            stack.enter_context(patch.object(runner.critic, 'train', checked_critic))
            actor_info, critic_info = runner.train()
        np.testing.assert_array_equal(cb.returns, returns_before)
        self.assertTrue(all(np.isfinite([float(v) for v in row.values()]).all()
                            for row in actor_info))
        self.assertTrue(np.isfinite([float(v) for v in critic_info.values()]).all())
        self.audit_year(runner, 1968)
        self.assertIsNot(previous_buffer, runner.critic_buffer)
        self.assertEqual(runner.sampled_years, [1965, 1968])
        self.assertEqual(runner.completed_episodes, 2)


if __name__ == '__main__':
    unittest.main()
