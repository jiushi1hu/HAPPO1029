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
from harl.common.valuenorm import ValueNorm

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
            _, _, reward, done, _, _ = env.step(rng.randint(200, size=5))
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
            env.step(np.full(5, 100))

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
            env.step(np.full(5, 100))
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

    def test_critic_normalized_mse_and_output_gradient(self):
        critic = self.r.critic
        self.assertFalse(critic.use_clipped_value_loss)
        self.assertFalse(critic.use_huber_loss)
        norm = ValueNorm(1)
        mean_acc, square_acc, weight = 0., 0., 0.
        for numbers in ([2.,4.,6.], [-5.,0.,8.], [3.,3.,3.]):
            targets = np.asarray(numbers, dtype=np.float32)[:,None]
            values = torch.tensor([[-2.],[0.],[2.]], requires_grad=True)
            before = targets.copy()
            beta = norm.beta
            mean_acc = beta*mean_acc + (1-beta)*targets.astype(np.float64).mean()
            square_acc = beta*square_acc + (1-beta)*(targets.astype(np.float64)**2).mean()
            weight = beta*weight + (1-beta)
            mean = mean_acc / max(weight, norm.epsilon)
            variance = max(square_acc/max(weight,norm.epsilon)-mean**2, .01)
            normalized = (targets-mean)/np.sqrt(variance)
            # With value clipping disabled, old predictions cannot change the loss.
            other_norm = copy.deepcopy(norm)
            loss = critic.cal_value_loss(values, torch.full_like(values, 1e6),
                                          torch.from_numpy(targets), norm)
            alternate = critic.cal_value_loss(values, torch.full_like(values, -1e6),
                                               torch.from_numpy(targets), other_norm)
            np.testing.assert_allclose(float(loss), .5*np.mean((values.detach().numpy()-normalized)**2),
                                       atol=1e-6, rtol=2e-6)
            torch.testing.assert_close(loss, alternate, rtol=0, atol=0)
            loss.backward()
            np.testing.assert_allclose(values.grad.numpy(), (values.detach().numpy()-normalized)/3,
                                       atol=1e-6, rtol=2e-6)
            actual_mean, actual_var = norm.running_mean_var()
            np.testing.assert_allclose(actual_mean.numpy(), mean, atol=1e-6, rtol=2e-6)
            np.testing.assert_allclose(actual_var.numpy(), variance, atol=1e-5, rtol=2e-6)
            np.testing.assert_allclose(norm.denormalize(norm.normalize(targets)), targets, atol=1e-6)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in norm.parameters()))
            np.testing.assert_array_equal(targets, before)
        # A constant batch must retain a finite normalization scale.
        constant = ValueNorm(1)
        constant.update(np.full((365,1), 3., dtype=np.float32))
        np.testing.assert_allclose(constant.running_mean_var()[1].numpy(), .01, atol=1e-8)
        self.assertTrue(torch.isfinite(constant.normalize(np.full((365,1),3.,dtype=np.float32))).all())

    def test_real_critic_updates_match_independent_mse_gradient_and_adam_steps(self):
        cfg, physical = load_configuration()
        r = CascadeReservoirRunner(self.data, physical, cfg, 1)
        reference = copy.deepcopy(r.critic)
        # Independent running-statistics state; do not call ValueNorm in the oracle.
        mean_acc, square_acc, weight = torch.zeros(1), torch.zeros(1), torch.tensor(0.)
        calls = []
        original = r.critic.update

        def update(sample, value_normalizer=None):
            nonlocal mean_acc, square_acc, weight
            self.assertIs(value_normalizer, r.value_normalizer)
            obs, old_values, returns, masks = sample
            ix = r.length-np.rint(obs[:,-1]*366).astype(int)
            np.testing.assert_array_equal(returns, saved_returns[ix,0])
            np.testing.assert_array_equal(old_values, saved_values[ix,0])
            self.assertEqual(len(ix), r.length)
            self.assertEqual(sorted(ix.tolist()), list(range(r.length)))
            targets = torch.from_numpy(returns)
            beta = value_normalizer.beta
            mean_acc = beta*mean_acc + (1-beta)*targets.mean(dim=0)
            square_acc = beta*square_acc + (1-beta)*targets.square().mean(dim=0)
            weight = beta*weight + (1-beta)
            mean = mean_acc/weight.clamp_min(value_normalizer.epsilon)
            variance = (square_acc/weight.clamp_min(value_normalizer.epsilon)-mean.square()).clamp_min(.01)
            normalized = (targets-mean)/variance.sqrt()
            values = reference.get_values(obs)
            expected_loss = .5*(values-normalized).square().mean()
            reference.critic_optimizer.zero_grad()
            # Supply the analytic derivative, independently of cal_value_loss().
            values.backward(reference.value_loss_coef*(values.detach()-normalized)/values.numel())
            parameters = list(reference.critic.parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
            expected_norm = torch.sqrt(sum(p.grad.square().sum() for p in parameters))
            scale = min(1., float(reference.max_grad_norm/(expected_norm+1e-6)))
            for p in parameters:
                p.grad.mul_(scale)
            reference.critic_optimizer.step()
            loss, grad_norm = original(sample, value_normalizer)
            torch.testing.assert_close(loss.detach(), expected_loss.detach(), atol=1e-6, rtol=2e-6)
            torch.testing.assert_close(grad_norm.detach(), expected_norm.detach(), atol=2e-6, rtol=2e-5)
            for actual, expected in zip(r.critic.critic.parameters(), parameters):
                torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
            for name, expected in [('running_mean',mean_acc),('running_mean_sq',square_acc),
                                   ('debiasing_term',weight)]:
                torch.testing.assert_close(getattr(value_normalizer,name),expected,atol=1e-7,rtol=2e-6)
            np.testing.assert_array_equal(r.critic_buffer.returns, saved_returns)
            np.testing.assert_array_equal(r.critic_buffer.value_preds, saved_values)
            for actor, state in zip(r.actor, actors_before):
                for key, value in state.items():
                    torch.testing.assert_close(actor.actor.state_dict()[key],value,rtol=0,atol=0)
            calls.append(float(loss))
            return loss, grad_norm

        for year in (1965,1968):
            r.collect_episode(year)
            saved_returns = r.critic_buffer.returns.copy()
            saved_values = r.critic_buffer.value_preds.copy()
            actors_before = [copy.deepcopy(a.actor.state_dict()) for a in r.actor]
            r.prep_training()
            start = len(calls)
            with patch.object(r.critic,'update',update):
                info = r.critic.train(r.critic_buffer,r.value_normalizer)
            self.assertEqual(len(calls)-start,cfg['algo']['critic_epoch'])
            self.assertAlmostEqual(info['value_loss'],np.mean(calls[start:]),places=6)
        # This exercises persistent optimizer moments as well as a fresh Adam state.
        self.assertTrue(all(int(state['step']) == len(calls)
                            for state in r.critic.critic_optimizer.state.values()))

    def test_clipped_actor_objective_and_log_probability_gradient(self):
        actor = self.r.actor[0]
        eps = actor.clip_param
        ratios = np.array([1+2*eps, 1-2*eps, 1+2*eps, 1-2*eps,
                           1+eps/2, 1-eps/2, 2., .1], dtype=np.float32)[:, None]
        advantages = np.array([2., 2., -2., -2., 3., -3., 1000., -1000.],
                              dtype=np.float32)[:, None]
        factor = np.array([2., .5, 3., 4., 1.5, .7, 9., 8.], dtype=np.float32)[:, None]
        active = np.array([1., 1., 1., 1., 1., 1., 0., 0.], dtype=np.float32)[:, None]
        lp = torch.tensor(np.log(ratios), requires_grad=True)
        entropy = torch.tensor(-.3, requires_grad=True)
        n = len(ratios)
        sample = (np.zeros((n,18), dtype=np.float32),
                  np.full((n,1), 100, dtype=np.int64),
                  np.ones((n,1), dtype=np.float32), active, np.zeros((n,1), dtype=np.float32),
                  advantages, None, factor)
        # Explicit cases: positive A clips only above the upper bound;
        # negative A clips only below the lower bound.
        effective = np.where(advantages >= 0, np.minimum(ratios, 1+eps),
                             np.maximum(ratios, 1-eps))
        expected_loss = -(active * factor * effective * advantages).sum() / active.sum()
        differentiable = np.where(advantages >= 0, ratios < 1+eps, ratios > 1-eps)
        expected_gradient = -active * factor * advantages * ratios * differentiable / active.sum()
        with patch.object(actor, 'evaluate_actions', return_value=(lp, entropy, None)), \
                patch.object(actor.actor_optimizer, 'step') as step:
            loss, _, _, actual_ratio = actor.update(sample)
        self.assertAlmostEqual(float(loss), float(expected_loss), places=5)
        np.testing.assert_allclose(actual_ratio.detach().numpy(), ratios, rtol=1e-6)
        np.testing.assert_allclose(lp.grad.numpy(), expected_gradient, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(float(entropy.grad), -actor.entropy_coef, places=7)
        step.assert_called_once()
        # Even an existing Adam state must not advance for an all-forced batch.
        forced_sample = sample[:3] + (np.zeros_like(active),) + sample[4:]
        with patch.object(actor, 'evaluate_actions') as evaluate, \
                patch.object(actor.actor_optimizer, 'step') as step:
            actor.update(forced_sample)
        evaluate.assert_not_called()
        step.assert_not_called()

    def test_actual_happo_epochs_keep_rollout_denominator_and_sequential_factors(self):
        from contextlib import ExitStack
        for order in ([4,2,0,3,1], [1,3,4,0,2]):
            with self.subTest(order=order):
                cfg, physical = load_configuration()  # Unchanged default network and epoch counts.
                r = CascadeReservoirRunner(self.data, physical, cfg, 1)
                r.collect_episode(1965)
                r.prep_training()
                old_stored = [b.action_log_probs.copy() for b in r.actor_buffer]

                def log_probs(i):
                    b = r.actor_buffer[i]
                    with torch.no_grad():
                        lp, _, _ = r.actor[i].evaluate_actions(b.obs[:-1,0],
                            b.actions[:,0], None, b.active_masks[:-1,0])
                    return lp.numpy().reshape(r.length,1,1)

                baseline = [log_probs(i).copy() for i in range(5)]
                expected_factor = np.ones((r.length,1,1), dtype=np.float64)
                completed, calls = [], [0]*5
                ratio_change = []
                critic_before = copy.deepcopy(r.critic.critic.state_dict())

                def wrap_update(i, original):
                    def update(sample):
                        obs, _, _, active, old_lp, adv, _, factor = sample
                        ix = r.length - np.rint(obs[:,-1]*366).astype(int)
                        np.testing.assert_array_equal(old_lp, old_stored[i][ix,0])
                        np.testing.assert_allclose(factor, expected_factor[ix,0], rtol=2e-5)
                        with torch.no_grad():
                            lp, _, _ = r.actor[i].evaluate_actions(sample[0], sample[1],
                                                                  sample[6], active)
                        ratio = np.exp(lp.numpy() - old_lp)
                        active_rows = active[:,0] > 0
                        if calls[i] == 0:
                            np.testing.assert_allclose(ratio[active_rows], 1., atol=1e-5, rtol=0)
                        else:
                            ratio_change.append(float(np.abs(ratio[active_rows]-1).max()))
                        effective = np.where(adv >= 0, np.minimum(ratio, 1+r.actor[i].clip_param),
                                             np.maximum(ratio, 1-r.actor[i].clip_param))
                        expected_loss = -(active*factor*effective*adv).sum()/active.sum()
                        result = original(sample)
                        self.assertAlmostEqual(float(result[0]), float(expected_loss), places=5)
                        calls[i] += 1
                        return result
                    return update

                def wrap_train(i, original):
                    def train(*args, **kwargs):
                        self.assertEqual(i, order[len(completed)])
                        np.testing.assert_allclose(r.actor_buffer[i].factor, expected_factor, rtol=2e-5)
                        others = [copy.deepcopy(a.actor.state_dict()) for a in r.actor]
                        result = original(*args, **kwargs)
                        for j, actor in enumerate(r.actor):
                            np.testing.assert_array_equal(r.actor_buffer[j].action_log_probs, old_stored[j])
                            if j != i:
                                for key, value in others[j].items():
                                    torch.testing.assert_close(actor.actor.state_dict()[key], value, rtol=0, atol=0)
                        for key, value in critic_before.items():
                            torch.testing.assert_close(r.critic.critic.state_dict()[key], value, rtol=0, atol=0)
                        completed_ratio = np.exp(log_probs(i).astype(np.float64)-baseline[i])
                        expected_factor[:] *= np.where(r.actor_buffer[i].active_masks[:-1] > 0,
                                                       completed_ratio, 1.)
                        completed.append(i)
                        return result
                    return train

                original_randperm = torch.randperm
                original_critic_train = r.critic.train

                def critic_train(*args, **kwargs):
                    self.assertEqual(completed, order)
                    return original_critic_train(*args, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(patch('torch.randperm', side_effect=lambda n, **kw:
                        torch.tensor(order) if n == 5 else original_randperm(n, **kw)))
                    for i, actor in enumerate(r.actor):
                        stack.enter_context(patch.object(actor, 'update', wrap_update(i, actor.update)))
                        stack.enter_context(patch.object(actor, 'train', wrap_train(i, actor.train)))
                    stack.enter_context(patch.object(r.critic, 'train', critic_train))
                    r.train()
                self.assertEqual(calls, [cfg['algo']['ppo_epoch']]*5)
                self.assertGreater(max(ratio_change), 1e-5)

    def test_sampling_400_years_with_replacement_is_reproducible(self):
        expected = np.random.RandomState(9).choice(self.data.training_years, size=400)
        actual = [self.r.sample_training_year() for _ in range(400)]
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(set(actual) <= set(self.data.training_years))
        self.assertLess(len(set(actual)), len(actual))
        for y in self.data.validation_years+self.data.test_years:
            self.assertNotIn(y, actual)

    def test_complete_annual_happo_update_factor_chain_and_index_log_probs(self):
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
            lp, _, _ = actor.evaluate_actions(b.obs[:-1].reshape(-1,18),
                                            b.actions.reshape(-1,1))
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
            lp, _, _ = actor.evaluate_actions(b.obs[:-1].reshape(-1,18),
                                            b.actions.reshape(-1,1))
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
