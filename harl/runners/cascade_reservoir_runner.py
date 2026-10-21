"""Reservoir rollout/evaluation adapter; HAPPO.train is inherited from HARL."""
import copy
import random
import time
import numpy as np
import torch

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.reward import EpisodeRewardMetrics, reward_settings
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.training_metrics import episode_record, log_episode
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


def validate_discount_factor(value):
    """Return a finite training discount in [0, 1], including episodic gamma=1."""
    message = "Discount factor gamma must be finite and between 0 and 1"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        gamma = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(gamma) or not 0. <= gamma <= 1.:
        raise ValueError(message)
    return gamma


def validate_gae_lambda(value):
    """Return a finite GAE trace parameter in [0, 1] for every entry point."""
    message = "GAE lambda must be finite and between 0 and 1"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        trace = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(trace) or not 0. <= trace <= 1.:
        raise ValueError(message)
    return trace


def validate_seed(value):
    """Use the integer seed range shared by all RNGs, including NumPy."""
    message = "Seed must be an integer between 0 and 4294967295"
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not 0 <= value <= 2**32-1):
        raise ValueError(message)
    return int(value)


def validate_fixed_training_year(value, data):
    """Restrict the optional in-sample diagnostic to the existing training split."""
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value not in data.training_years):
        raise ValueError('fixed_training_year must be an integer from the training split; '
                         'validation and test years are reserved')
    return int(value)


def configure_calendar(algo_args, data, year=None):
    """Allocate a complete sampled year without padding or dropping leap days."""
    args = copy.deepcopy(algo_args)
    args["algo"]["gamma"] = validate_discount_factor(args["algo"]["gamma"])
    args["algo"]["gae_lambda"] = validate_gae_lambda(args["algo"]["gae_lambda"])
    if args["train"]["n_rollout_threads"] != 1:
        raise ValueError("Annual sampling uses one environment: one year = one episode")
    year = data.training_years[0] if year is None else int(year)
    if year not in data.training_years:
        raise ValueError("Only training years may be sampled")
    start, stop = data.indices(year, year)
    args["train"]["horizon_days"] = stop - start
    args["train"]["episode_length"] = stop - start
    args["train"]["sampled_year"] = year
    args["train"].pop("rollout_days", None)
    for key in ("actor_num_mini_batch", "critic_num_mini_batch"):
        requested = int(args["algo"][key])
        if requested < 1:
            raise ValueError("Minibatch counts must be positive")
        args["algo"][key] = max(n for n in range(1, min(requested, stop - start) + 1)
                                if (stop - start) % n == 0)
    return args


class CascadeReservoirRunner(OnPolicyHARunner):
    """Specialized calendar/physics integration with the existing optimizer stack.

    Does not invoke the generic game-environment factory or auto-reset wrapper.
    All normalized actions see the same beginning-of-day full observation;
    the physical environment resolves releases upstream to downstream.
    """

    def __init__(self, data, physical_config, algo_args, seed, episode_log=None):
        seed = validate_seed(seed)
        self.reward_settings = reward_settings(physical_config)
        distribution = algo_args['model'].get('action_distribution')
        if distribution != 'categorical':
            raise ValueError('Cascade controls require categorical discrete actions')
        self.action_bins = algo_args['model'].get('action_bins', 200)
        for key in ("use_policy_active_masks", "mask_inactive_factors"):
            if not algo_args["algo"].get(key, False):
                raise ValueError("Forced cascade actions require " + key)
        self.data, self.physical_config = data, physical_config
        self.fixed_training_year = validate_fixed_training_year(
            algo_args['experiment'].get('fixed_training_year'), data)
        self.requested_args = copy.deepcopy(algo_args)
        self.algo_args = algo_args = configure_calendar(algo_args, data, self.fixed_training_year)
        self.training_years = data.training_years
        self.year_rng = np.random.RandomState(seed)
        self.sampled_years = []
        self.seed = seed
        self.num_agents, self.state_type = 5, "EP"
        self.share_param = False
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.action_aggregation = "prod"
        self.device = torch.device("cpu")
        self.value_normalizer = ValueNorm(1, device=self.device) if algo_args["train"]["use_valuenorm"] else None
        torch.set_num_threads(algo_args["device"]["torch_threads"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.n_envs = algo_args["train"]["n_rollout_threads"]
        self.length = algo_args["train"]["episode_length"]
        if algo_args["algo"]["share_param"]:
            raise ValueError("The cascade specification requires independent actors")
        for key in ("actor_num_mini_batch", "critic_num_mini_batch"):
            if self.length * self.n_envs % algo_args["algo"][key]:
                raise ValueError("Choose minibatch count dividing rollout size; avoid dropping samples")
        initial_year = self.training_years[0] if self.fixed_training_year is None else self.fixed_training_year
        self.envs = [CascadeReservoirEnv(data, physical_config, start_year=initial_year,
                                       end_year=initial_year, action_bins=self.action_bins)]
        env = self.envs[0]
        network_args = {**algo_args["model"], **algo_args["algo"]}
        self.actor = [HAPPO(network_args, env.observation_space[i], env.action_space[i], self.device) for i in range(5)]
        self.critic = VCritic(network_args, env.share_observation_space[0], self.device)
        self._allocate_buffers()
        self.iteration, self.training_steps = 0, 0
        self.best_energy = -float("inf")
        self.best_score = -float("inf")
        self.best_iteration = None
        self.started = time.monotonic()
        self.last_rollout = {}
        self.last_reward_metrics = EpisodeRewardMetrics().summary()
        self.completed_episodes = 0
        self.validation_history = []
        self.fixed_year_history = []
        self.training_history = []
        self.episode_history = []
        self.episode_log = episode_log
        self.episode_attempts = 0
        self.hard_failed_episodes = 0
        self.best_actor_states = None
        self.diagnostics_enabled = bool(algo_args["experiment"].get("diagnostics", False))
        self.diagnostics_history = []
        self.validation_diagnostics = []

    def _allocate_buffers(self):
        args, env = self.algo_args, self.envs[0]
        self.actor_buffer = [OnPolicyActorBuffer({**args["train"], **args["model"]},
                             env.observation_space[i], env.action_space[i]) for i in range(5)]
        self.critic_buffer = OnPolicyCriticBufferEP(
            {**args["train"], **args["model"], **args["algo"]}, env.share_observation_space[0])
        for actor in self.actor:
            actor.actor_num_mini_batch = args["algo"]["actor_num_mini_batch"]
        self.critic.critic_num_mini_batch = args["algo"]["critic_num_mini_batch"]

    def sample_training_year(self):
        # Independent of action sampling, optimizer shuffling and evaluation.
        if self.fixed_training_year is not None:
            return self.fixed_training_year
        return int(self.year_rng.choice(self.training_years))

    def _log_episode(self, rewards, energy, failure=None):
        self.hard_failed_episodes += int(failure is not None)
        record = episode_record(self.envs[0], rewards, energy, self.seed, self.episode_attempts,
            self.algo_args['algo']['gamma'], self.algo_args['algo']['gae_lambda'],
            self.hard_failed_episodes, failure)
        self.episode_history.append(record)
        log_episode(record, self.episode_log)

    @torch.no_grad()
    def collect_episode(self, year=None):
        """Collect a complete sampled or fixed training year from initial storage."""
        if (self.fixed_training_year is not None and year is not None and
                year != self.fixed_training_year):
            raise ValueError('Fixed-year diagnostic cannot collect a different training year')
        year = self.sample_training_year() if year is None else int(year)
        self.algo_args = configure_calendar(self.requested_args, self.data, year)
        self.length = self.algo_args["train"]["episode_length"]
        self.envs = [CascadeReservoirEnv(self.data, self.physical_config,
                                       start_year=year, end_year=year, action_bins=self.action_bins)]
        self.episode_attempts += 1
        self._allocate_buffers()
        self.prep_rollout()
        obs = np.stack([env.observation() for env in self.envs])
        for b in self.actor_buffer:
            b.step = 0; b.obs[0] = obs
            b.masks.fill(1); b.active_masks.fill(1)
        cb = self.critic_buffer
        cb.step = 0; cb.share_obs[0] = obs
        cb.masks.fill(1); cb.bad_masks.fill(1)
        energy, forced, largest_residual = np.zeros(self.n_envs), 0, 0.
        reward_metrics = EpisodeRewardMetrics()
        energy_by_reservoir = np.zeros(5)
        for t in range(self.length):
            actions, log_probs = [], []
            for i in range(5):
                b = self.actor_buffer[i]
                a, lp = self.actor[i].get_actions(b.obs[t])
                actions.append(_t2n(a)); log_probs.append(_t2n(lp))
            actions = np.stack(actions, axis=1)
            values = self.critic.get_values(cb.share_obs[t])
            for e, env in enumerate(self.envs):
                try:
                    env.step(actions[e])
                except (InfeasibleCascade, AssertionError) as exc:
                    self._log_episode(reward_metrics, energy_by_reservoir,
                        dict(date=env.data.dates[env.index].isoformat(),
                             error_type=type(exc).__name__, reason=str(exc)))
                    raise  # Observability only: preserve strict failure behavior.
                energy[e] += env.last_transition["energy"].sum()
                energy_by_reservoir += env.last_transition['energy']
                reward_metrics.update(env.last_transition)
            results = [env.last_transition for env in self.envs]
            done = np.array([env.index == env.stop for env in self.envs])
            next_obs = np.stack([env.observation() for env in self.envs])
            controls = np.stack([r["controls"] for r in results])
            effective_actions = np.stack([r["actions"] for r in results])
            forced += int((controls == 0).sum())
            largest_residual = max(largest_residual, max(float(np.abs(r["residual"]).max()) for r in results))
            next_masks = (~done).astype(np.float32)[:, None]
            for i, b in enumerate(self.actor_buffer):
                lp = np.where(controls[:, i:i+1] > 0, log_probs[i], 0.)
                b.insert(next_obs, effective_actions[:, i:i+1], lp, next_masks,
                         np.ones((self.n_envs, 1), dtype=np.float32))
                # Controllability belongs to the sampled action at t, not t+1.
                b.active_masks[t] = controls[:, i:i+1]
            rewards = np.array([[r["reward"]] for r in results], dtype=np.float32)
            cb.insert(next_obs, _t2n(values), rewards, next_masks,
                      np.ones_like(next_masks))
        self.completed_episodes += 1
        self.last_reward_metrics = reward_metrics.summary()
        self._log_episode(reward_metrics, energy_by_reservoir)
        self.sampled_years.append(year)
        self.last_rollout = dict(sampled_year=year, episode_days=self.length,
                                 episode_energy_TWh=float(energy.mean() / 1e6),
                                 episode_mean_daily_energy_MWh=float(energy.mean() / self.length),
                                 completed_episodes=self.completed_episodes,
                                 forced_fraction=forced / (self.length * self.n_envs * 5),
                                 max_balance_residual_m3=largest_residual)
        self.compute()  # Reuse OnPolicyBaseRunner and original GAE buffer.
        predictions = cb.value_preds[:-1]
        if self.value_normalizer is not None:
            predictions = self.value_normalizer.denormalize(predictions)
        targets = cb.returns[:-1]
        # These measure agreement with GAE targets, not Monte Carlo ground truth.
        self.last_rollout.update(
            gae_target_mean=float(targets.mean()),
            value_prediction_mean=float(predictions.mean()),
            gae_target_rmse=float(np.sqrt(np.mean((targets - predictions) ** 2))),
            gae_target_explained_variance=float(1. - np.var(targets - predictions) /
                                               max(float(np.var(targets)), 1e-8)))

    def evaluate_checkpoint(self):
        """Keep in-sample diagnostic checkpoints separate from held-out validation."""
        if self.fixed_training_year is not None:
            return self.evaluate_training_year()
        return self.validate()

    def evaluate_training_year(self):
        """Evaluate the current actors on a fresh copy of the fixed training year.

        Evaluation uses categorical modes, never adds samples to the training
        buffers, and does not invoke multi-year/stochastic validation reports.
        """
        if self.fixed_training_year is None:
            raise RuntimeError('A fixed training year is required for this diagnostic')
        with torch.random.fork_rng(devices=[]):
            row = evaluate_policy(self.actor, self.data, self.physical_config,
                self.fixed_training_year, self.seed, self.iteration, self.training_steps,
                deterministic=True, diagnostics=True)
        power = float(row['energy_MWh']/self.envs[0].physics.reference_energy)
        terms = dict(band='target_band', rise='daily_rise', fall='daily_fall')
        costs = {name: float(row.get('episode_penalty_'+term, 0.)) for name, term in terms.items()}
        total = float(row.get('episode_reward', power))
        if (not np.isfinite([power, total]+list(costs.values())).all() or
                not np.isclose(total, power-sum(costs.values()), rtol=1e-9, atol=1e-9)):
            raise ValueError('Invalid fixed-year evaluation reward components')
        if row['feasible'] is not True or row['violation_count'] != 0:
            raise ValueError('Infeasible fixed-year evaluation')
        percentages = {term: 100.*costs[name]/power if power > 0. else None
                       for name, term in terms.items()}
        percentages['total'] = 100.*sum(costs.values())/power if power > 0. else None
        record = dict(record_type='fixed_year_evaluation', seed=self.seed,
            episode=self.iteration, training_steps=self.training_steps,
            year=self.fixed_training_year, deterministic=True,
            actual_generation_TWh=float(row['energy_TWh']), power_reward=power,
            total_reward=total, total_soft_penalty=sum(costs.values()),
            **{name+'_penalty': value for name, value in costs.items()},
            penalty_to_power_percent=percentages,
            gamma=float(self.algo_args['algo']['gamma']),
            gae_lambda=float(self.algo_args['algo']['gae_lambda']),
            reward_penalties=dict(self.reward_settings),
            feasible=row['feasible'], violation_count=row['violation_count'],
            evaluation=row)
        if self.diagnostics_enabled and self.diagnostics_history:
            # These critic/actor statistics belong to the latest sampled
            # training rollout, separately from this deterministic evaluation.
            record['last_training_update_diagnostics'] = self.diagnostics_history[-1]
        self.fixed_year_history.append(record)
        score = total if self.reward_settings['enabled'] else float(row['energy_MWh'])
        if score > self.best_score:
            self.best_score, self.best_energy, self.best_iteration = score, float(row['energy_MWh']), self.iteration
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        log_episode(record, self.episode_log)
        print('seed={} episode={} fixed_training_year={} deterministic_generation={:.3f} TWh '
              'power={:.6f} band={:.6f} rise={:.6f} fall={:.6f} total={:.6f} '
              '(in-sample diagnostic)'.format(self.seed, self.iteration, self.fixed_training_year,
                  row['energy_TWh'], power, costs['band'], costs['rise'], costs['fall'], total), flush=True)
        return score

    def validate(self):
        rows = [evaluate_policy(self.actor, self.data, self.physical_config, year,
                                self.seed, self.iteration, self.training_steps,
                                **({"diagnostics": True} if self.diagnostics_enabled else {}))
                for year in self.data.validation_years]
        energy = float(np.mean([row["energy_MWh"] for row in rows]))
        score = float(np.mean([row['episode_reward'] for row in rows])) if self.reward_settings['enabled'] else energy
        if not np.isfinite(energy) or not np.isfinite(score):
            raise FloatingPointError('Non-finite validation score')
        history = dict(episode=self.iteration, training_steps=self.training_steps,
                       mean_TWh=energy/1e6, years=rows)
        if self.reward_settings['enabled']:
            history.update(mean_reward=score,mean_penalty=float(np.mean([row['episode_penalty'] for row in rows])))
        self.validation_history.append(history)
        if score > self.best_score:
            self.best_score, self.best_energy, self.best_iteration = score, energy, self.iteration
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        if self.reward_settings['enabled']:
            print('seed={} episode={} steps={} validation_2020_2024_mean={:.3f} TWh reward={:.6f} '
                  'penalty={:.6f} best_reward={:.6f} selected_generation={:.3f} TWh'.format(
                self.seed,self.iteration,self.training_steps,energy/1e6,score,history['mean_penalty'],
                self.best_score,self.best_energy/1e6),flush=True)
        else:
            print("seed={} episode={} steps={} validation_2020_2024_mean={:.3f} TWh best={:.3f} TWh".format(
                self.seed, self.iteration, self.training_steps, energy / 1e6, self.best_energy / 1e6), flush=True)
        if self.diagnostics_enabled:
            milestones = self.algo_args["experiment"].get("diagnostic_eval_episodes", [0,20,100,150,400])
            if self.iteration in milestones or self.iteration == self.algo_args["experiment"]["episodes"]:
                from harl.envs.cascade_reservoir.diagnostics import validation_report
                self.validation_diagnostics.append(validation_report(self, rows))
        return score

    def run(self, episodes, eval_interval):
        if episodes < 1 or eval_interval < 1:
            raise ValueError("Positive training episodes/evaluation interval required")
        self.evaluate_checkpoint()  # Honest untrained point on the learning curve.
        for k in range(1, episodes + 1):
            self.iteration = k
            if self.algo_args["train"]["use_linear_lr_decay"]:
                for actor in self.actor:
                    actor.lr_decay(k, episodes)
                self.critic.lr_decay(k, episodes)
            self.collect_episode()
            diagnostic_before = None
            if self.diagnostics_enabled:
                from harl.envs.cascade_reservoir.diagnostics import before_update
                diagnostic_before = before_update(self)
            self.prep_training()
            actor_info, critic_info = super().train()  # Original sequential HAPPO implementation.
            self.training_steps += self.length * self.n_envs
            row = dict(seed=self.seed, iteration=k, training_steps=self.training_steps,
                       elapsed_seconds=time.monotonic() - self.started, **self.last_rollout,
                       **self.last_reward_metrics)
            for i, info in enumerate(actor_info):
                for name, value in info.items():
                    row["actor{}_{}".format(i, name)] = float(value)
            row.update({"critic_" + name: float(v) for name, v in critic_info.items()})
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError("Non-finite training diagnostics")
            self.training_history.append(row)
            if self.diagnostics_enabled:
                from harl.envs.cascade_reservoir.diagnostics import after_update, print_update_window
                self.diagnostics_history.append(after_update(self, diagnostic_before))
                if k % eval_interval == 0 or k == episodes:
                    print_update_window(self, min(eval_interval,k))
            if k % eval_interval == 0 or k == episodes:
                self.evaluate_checkpoint()
        if self.fixed_training_year is not None:
            result = dict(seed=self.seed, fixed_training_year=self.fixed_training_year,
                          best_episode=self.best_iteration, best_training_year_TWh=self.best_energy/1e6)
            if self.reward_settings['enabled']:
                result['best_training_year_reward'] = self.best_score
            return result
        result = dict(seed=self.seed, best_episode=self.best_iteration, best_iteration=self.best_iteration,
                      best_validation_TWh=self.best_energy / 1e6)
        if self.reward_settings['enabled']:
            result['best_validation_reward'] = self.best_score
        return result

    def final_test(self):
        """Evaluate selected actors once on each user-defined typical year."""
        if self.fixed_training_year is not None:
            raise RuntimeError('Fixed-training-year diagnostics do not run held-out tests')
        if self.best_actor_states is None:
            raise RuntimeError("No validation-selected policy is available")
        for actor, weights in zip(self.actor, self.best_actor_states):
            actor.actor.load_state_dict(weights)
        return [dict(evaluate_policy(self.actor, self.data, self.physical_config, year,
                                    self.seed, self.best_iteration, self.training_steps),
                     typical_label=label)
                for year, label in zip(self.data.test_years, self.data.test_labels)]
