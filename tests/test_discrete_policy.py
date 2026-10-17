"""Categorical action probabilities and complete annual return targets."""
from pathlib import Path
import unittest

import numpy as np
import torch
from gym.spaces import Box, Discrete

from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.models.base.act import ACTLayer
from harl.models.base.distributions import FixedCategorical, FixedNormal
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class DiscretePolicyTests(unittest.TestCase):
    def test_sampling_likelihoods_argmax_and_gradients(self):
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo']}
        torch.manual_seed(3)
        layer = ACTLayer(Discrete(200), 4, 'orthogonal_', .01, args)
        features = torch.randn(100, 4)
        actions, old_log_probs = layer(features)
        log_probs, entropy, dist = layer.evaluate_actions(features, actions)
        self.assertIsInstance(dist, FixedCategorical)
        self.assertEqual(dist.probs.shape, (100, 200))
        self.assertEqual(actions.dtype, torch.int64)
        self.assertTrue(((actions >= 0) & (actions < 200)).all())
        torch.testing.assert_close(log_probs, dist.probs.gather(1, actions).log())
        torch.testing.assert_close((log_probs - old_log_probs).exp(), torch.ones_like(log_probs))
        torch.testing.assert_close(layer(features, deterministic=True)[0],
                                   dist.probs.argmax(-1, keepdim=True))
        (-log_probs.mean() - .001 * entropy).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in layer.parameters()))

    def test_generic_gaussian_action_layer_is_unchanged(self):
        cfg, _ = load_configuration()
        args = {**cfg['model'], **cfg['algo']}
        layer = ACTLayer(Box(-1., 1., (1,), dtype=np.float32), 4, 'orthogonal_', .01, args)
        self.assertIsInstance(layer.action_out(torch.zeros(1, 4)), FixedNormal)

    def test_lambda_one_uses_selected_discount_without_critic_bootstrap(self):
        cfg, physical = load_configuration()
        cfg['algo']['gae_lambda'] = 1.
        cfg['model']['hidden_sizes'] = [16, 16]
        data = CascadeData(Path(__file__).resolve().parents[1] / 'data/cascade_reservoir_new',
                           physical, cfg['experiment'])
        for gamma, year in ((0., 1965), (.99, 1968), (1., 1965)):
            with self.subTest(gamma=gamma, year=year):
                cfg['algo']['gamma'] = gamma
                runner = CascadeReservoirRunner(data, physical, cfg, 1)
                runner.value_normalizer.update(np.array([[20.], [40.]], dtype=np.float32))
                runner.collect_episode(year)
                cb = runner.critic_buffer
                self.assertEqual(cb.gamma, gamma)
                rewards = cb.rewards.astype(np.float64)
                # Independent forward weighted sum, including all leap-year days.
                expected = np.stack([(rewards[t:] *
                    (gamma ** np.arange(len(rewards)-t))[:, None, None]).sum(axis=0)
                    for t in range(len(rewards))])
                np.testing.assert_allclose(cb.returns[:-1], expected, rtol=1e-5, atol=1e-4)
                np.testing.assert_allclose(runner.envs[0].storage, data.initial_storage(year),
                                           rtol=0., atol=1e-3)


if __name__ == '__main__':
    unittest.main()
