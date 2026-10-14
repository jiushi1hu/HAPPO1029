"""Quadratic physics and independent-actor HAPPO integration checks."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.optimize import linprog
from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

ROOT = Path(__file__).resolve().parents[1]


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.physical = load_configuration()
        cls.data = CascadeData(ROOT / "data/cascade_reservoir_new", cls.physical, cls.cfg["experiment"])
        cls.physics = ReservoirPhysics(cls.data, cls.physical)

    def test_quadratic_mapping_endpoints_and_midpoint(self):
        v = .5*(self.data.vmin+self.data.vmax)
        ext = self.data.external[0]
        for u, fraction in [(0., 0.), (.5, .25), (1., 1.)]:
            r = self.physics.transition(v, ext, np.full(5, u))
            np.testing.assert_allclose(r["release"], r["low"]+fraction*(r["high"]-r["low"]))
            np.testing.assert_array_equal(r["actions"], np.full(5, u))

    def test_complete_test_year_conserves_water_and_uses_same_day_routing(self):
        d = self.data
        env = CascadeReservoirEnv(d, self.physical, start_year=1998, end_year=1998)
        rng = np.random.RandomState(71)
        incoming, outgoing = 0., 0.
        for _ in range(env.stop-env.start):
            _, _, reward, done, _, _ = env.step(rng.uniform(size=5))
            r = env.last_transition
            incoming += r["external"].sum()*86400
            outgoing += r["release"][-1]*86400
            self.assertTrue((env.storage >= d.vmin-1.).all())
            self.assertTrue((env.storage <= d.vmax+1.).all())
            np.testing.assert_allclose(r["inflow"][1:], r["release"][:-1]+r["external"][1:])
            self.assertAlmostEqual(float(reward[0, 0]), r["energy"].sum()/1624800., places=6)
        self.assertTrue(done.all())
        self.assertLess(abs(env.storage.sum()-d.initial_storage(1998).sum()-incoming+outgoing), .1)
        with self.assertRaises(RuntimeError):
            env.step(np.full(5, .5))

    def test_intervals_against_independent_linear_program(self):
        d = self.data; rng = np.random.RandomState(12)
        matrix = np.eye(5); matrix[np.arange(1, 5), np.arange(4)] = -1.
        for _ in range(8):
            v = rng.uniform(d.vmin, d.vmax); ext = d.external[rng.randint(len(d.dates))]
            lo_rhs = ext+(v-d.vmax)/86400; hi_rhs = ext+(v-d.vmin)/86400
            a = np.vstack([matrix, -matrix]); b = np.concatenate([hi_rhs, -lo_rhs])
            bounds = list(zip(np.zeros(5), d.release_cap))
            r = self.physics.transition(v, ext, rng.uniform(size=5))
            for i in range(5):
                c = np.eye(5)[i]
                lo = linprog(c, A_ub=a, b_ub=b, bounds=bounds, method="highs")
                hi = linprog(-c, A_ub=a, b_ub=b, bounds=bounds, method="highs")
                self.assertTrue(lo.success and hi.success)
                self.assertAlmostEqual(lo.fun, r["low"][i], places=6)
                self.assertAlmostEqual(-hi.fun, r["high"][i], places=6)
                bounds[i] = (r["release"][i], r["release"][i])

    def test_forced_actions_and_infeasibility(self):
        r = self.physics.transition(self.data.vmin, np.zeros(5), np.arange(5)/5.)
        np.testing.assert_array_equal(r["controls"], np.zeros(5))
        np.testing.assert_array_equal(r["release"], np.zeros(5))
        with self.assertRaises(InfeasibleCascade):
            self.physics.transition(self.data.vmax, np.full(5, 1e7), np.full(5, .5))

    def test_initial_state_is_selected_by_year_and_leap_day(self):
        d = copy.deepcopy(self.data)
        d.initial_storage_by_year[1968] = d.vmin.copy()
        env = CascadeReservoirEnv(d, self.physical, start_year=1968, end_year=1968)
        np.testing.assert_array_equal(env.storage, d.vmin)
        for _ in range(59):
            env.step(np.full(5, .5))
        self.assertEqual(d.dates[env.index].isoformat(), "1968-02-29")
        self.assertAlmostEqual(float(env.observation()[-1]), 307/366., places=6)


class LearningTests(unittest.TestCase):
    def setUp(self):
        cfg, self.physical = load_configuration()
        cfg["model"]["hidden_sizes"] = [16, 16]
        for key in ["ppo_epoch", "critic_epoch", "actor_num_mini_batch", "critic_num_mini_batch"]:
            cfg["algo"][key] = 1
        self.data = CascadeData(ROOT / "data/cascade_reservoir_new", self.physical, cfg["experiment"])
        self.r = CascadeReservoirRunner(self.data, self.physical, cfg, 9)

    def test_sampling_400_years_with_replacement_is_reproducible(self):
        expected = np.random.RandomState(9).choice(self.data.training_years, size=400)
        actual = [self.r.sample_training_year() for _ in range(400)]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(set(actual) <= set(self.data.training_years))
        self.assertLess(len(set(actual)), len(actual))
        for y in self.data.validation_years+self.data.test_years:
            self.assertNotIn(y, actual)

    def test_complete_annual_happo_update_factor_chain_and_latent_log_probs(self):
        r = self.r; r.collect_episode(1965); r.prep_training()
        self.assertEqual(r.length, 365)
        ids = [set(id(p) for p in a.actor.parameters()) for a in r.actor]
        self.assertTrue(all(not ids[i]&ids[j] for i in range(5) for j in range(i)))
        # Float32 GAE subtracts/adds the value prediction; terminal rewards can
        # now be close to zero when closure leaves no discretionary release.
        np.testing.assert_allclose(r.critic_buffer.returns[-2], r.critic_buffer.rewards[-1], atol=1e-7)
        self.assertEqual(r.critic_buffer.masks[-1, 0, 0], 0.)
        old = []
        for actor, b in zip(r.actor, r.actor_buffer):
            lp, _, _ = actor.evaluate_actions(b.obs[:-1].reshape(-1,18), b.rnn_states[0],
                                            b.actions.reshape(-1,1), b.masks[:-1].reshape(-1,1))
            old.append(lp.detach().numpy())
            active = b.active_masks[:-1].reshape(-1) > 0
            np.testing.assert_allclose(old[-1][active], b.action_log_probs.reshape(-1,1)[active], atol=1e-6)
            np.testing.assert_array_equal(b.action_log_probs.reshape(-1,1)[~active], 0.)
        np.testing.assert_allclose(r.envs[0].storage, self.data.initial_storage(1965), rtol=0, atol=1e-3)
        order = [4, 2, 0, 3, 1]; original = torch.randperm
        with patch("torch.randperm", side_effect=lambda n, **kw: torch.tensor(order) if n==5 else original(n, **kw)):
            infos, critic = r.train()
        factor = np.ones((365,1,1))
        for i in order:
            actor, b = r.actor[i], r.actor_buffer[i]
            np.testing.assert_allclose(b.factor, factor, rtol=2e-5)
            lp, _, _ = actor.evaluate_actions(b.obs[:-1].reshape(-1,18), b.rnn_states[0],
                                            b.actions.reshape(-1,1), b.masks[:-1].reshape(-1,1))
            factor *= np.where(b.active_masks[:-1]>0, np.exp(lp.detach().numpy()-old[i]).reshape(365,1,1), 1.)
        self.assertTrue(np.isfinite(float(critic["value_loss"])))
        self.assertGreater(float(r.value_normalizer.debiasing_term), 0.)

    def test_mean_validation_selects_actors_and_test_years_used_only_at_end(self):
        r = self.r
        def fake(actors, data, physical, year, seed, episode, steps):
            return dict(year=year, seed=seed, energy_MWh=year*1e6, energy_TWh=float(year))
        target = "harl.runners.cascade_reservoir_runner.evaluate_policy"
        with patch(target, side_effect=fake) as f:
            score = r.validate()
            self.assertEqual([c.args[3] for c in f.call_args_list], list(self.data.validation_years))
            self.assertEqual(score/1e6, 2022.)
            self.assertEqual(r.best_iteration, 0)
            f.reset_mock(); tests = r.final_test()
            self.assertEqual([c.args[3] for c in f.call_args_list], list(self.data.test_years))
            self.assertEqual([x["typical_label"] for x in tests], list(self.data.test_labels))

    def test_real_validation_is_frozen_and_reproducible(self):
        before = [{k:v.clone() for k,v in a.actor.state_dict().items()} for a in self.r.actor]
        a = evaluate_policy(self.r.actor, self.data, self.physical, 2024, 9, 0, 0)
        b = evaluate_policy(self.r.actor, self.data, self.physical, 2024, 9, 0, 0)
        self.assertEqual(a, b)
        for actor, state in zip(self.r.actor, before):
            for key in state:
                torch.testing.assert_close(state[key], actor.actor.state_dict()[key], rtol=0, atol=0)
        for i,name in enumerate(IDS):
            self.assertEqual(a[name+"_start_m3"], self.data.initial_storage(2024)[i])

    def test_two_episodes_have_exact_count_and_reinitialize_each_year(self):
        r = self.r
        with patch.object(r, "sample_training_year", side_effect=[1968, 1965]), patch.object(r, "validate", return_value=0.):
            r.run(2, 1)
        self.assertEqual(r.completed_episodes, 2)
        self.assertEqual(r.training_steps, 731)
        self.assertEqual(r.sampled_years, [1968, 1965])
        np.testing.assert_allclose(r.critic_buffer.share_obs[0,0,:5], 1.)


if __name__ == "__main__":
    unittest.main()
