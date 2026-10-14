"""Runner options must have the same execution semantics as the HARL runner."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner

ROOT = Path(__file__).resolve().parents[1]


class RunnerConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.cfg['model']['hidden_sizes'] = [16, 16]
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.cfg['experiment'])

    def test_fixed_order_reaches_actual_actor_updates(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['algo']['fixed_order'] = True
        r = CascadeReservoirRunner(self.data, self.physical, cfg, 5)
        self.assertTrue(r.fixed_order)
        r.collect_episode(1965)
        r.prep_training()
        seen = []
        for i, actor in enumerate(r.actor):
            original = actor.train
            def spy(*args, _i=i, _original=original, **kwargs):
                seen.append(_i)
                return _original(*args, **kwargs)
            actor.train = spy
        r.train()
        self.assertEqual(seen, list(range(5)))

    def test_actor_and_critic_learning_rates_follow_decay_option(self):
        for enabled in [False, True]:
            with self.subTest(decay=enabled):
                cfg = copy.deepcopy(self.cfg)
                cfg['train']['use_linear_lr_decay'] = enabled
                r = CascadeReservoirRunner(self.data, self.physical, cfg, 5)
                rates = []
                def fake_train(_):
                    rates.append([a.actor_optimizer.param_groups[0]['lr'] for a in r.actor] +
                                 [r.critic.critic_optimizer.param_groups[0]['lr']])
                    return [{} for _ in range(5)], {}
                with patch.object(r, 'validate', return_value=0.), \
                     patch.object(r, 'collect_episode'), \
                     patch.object(OnPolicyHARunner, 'train', new=fake_train), \
                     redirect_stdout(io.StringIO()):
                    r.run(2, 1)
                initial = np.array([cfg['model']['lr']]*5 + [cfg['model']['critic_lr']])
                np.testing.assert_allclose(rates[0], initial, rtol=1e-12)
                np.testing.assert_allclose(rates[1], initial*(.5 if enabled else 1.), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
