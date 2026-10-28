"""Reservoir rollout/evaluation adapter; HAPPO.train is inherited from HARL."""
import copy
import random
import time
import numpy as np
import torch

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic, validate_value_normalization
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.valuenorm import ValueNorm
from harl.envs.cascade_reservoir.cascade_reservoir_env import CascadeReservoirEnv
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, reward_settings, validate_discount_factor, REWARD_AGGREGATION,
    PENALTY_TERMS, REWARD_DEFINITION)
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.training_metrics import episode_record, log_episode
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


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


def validate_rollout_episodes(value):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError('rollout_episodes must be a positive integer')
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
    validate_value_normalization(args['algo']['use_clipped_value_loss'], args['train']['use_valuenorm'])
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
        requested = args['algo'][key]
        if (isinstance(requested, (bool, np.bool_)) or
                not isinstance(requested, (int, np.integer)) or requested < 1):
            raise ValueError('Minibatch counts must be positive integers')
        requested = int(requested)
        args["algo"][key] = max(n for n in range(1, min(requested, stop - start) + 1)
                                if (stop - start) % n == 0)
    return args


class CascadeReservoirRunner(OnPolicyHARunner):
    """Specialized calendar/physics integration with the existing optimizer stack.

    Does not invoke the generic game-environment factory or auto-reset wrapper.
    Each actor sees its own beginning-of-day local/neighbor observation. The
    critic sees the centralized state; physics resolves releases upstream to downstream.
    """
    collection_mode = 'annual'

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
        self.validation_enabled = bool(getattr(data, 'validation_years', ()))
        self.requested_args = copy.deepcopy(algo_args)
        self.rollout_episodes = validate_rollout_episodes(
            algo_args['experiment'].get('rollout_episodes', 1))
        self.algo_args = algo_args = configure_calendar(algo_args, data, self.fixed_training_year)
        self.training_years = data.training_years
        self.year_rng = np.random.RandomState(seed)
        self.sampled_years = []
        self.seed = seed
        self.num_agents = 5
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
        self.update_count = 0
        self.joint_kl_info = None
        self.collected_training_steps = 0
        self.batch_episode_starts = np.array([0], dtype=int)
        self.batch_years = []
        self.batch_episode_lengths = []
        self.best_energy = -float("inf")
        self.best_score = -float("inf")
        self.best_iteration = None
        self.best_training_steps = None
        self.best_update = None
        self.started = time.monotonic()
        self.last_rollout = {}
        self.last_reward_metrics = EpisodeRewardMetrics(algo_args['algo']['gamma']).summary()
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
        self.fixed_year_diagnostics = []

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
        record.update(rollout_episodes_per_update=self.rollout_episodes,
                      collecting_for_update=self.update_count+1,
                      policy_updates_completed=self.update_count,
                      ppo_epochs_max=self.requested_args['algo']['ppo_epoch'],
                      actor_target_kl=self.requested_args['algo'].get('target_kl'),
                      actor_kl_rollback=self.requested_args['algo'].get('kl_rollback', False),
                      joint_target_kl=self.requested_args['algo'].get('joint_target_kl'))
        self.collected_training_steps += record['simulated_days']
        record['training_steps'] = self.collected_training_steps
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
        self.batch_episode_starts = np.array([0],dtype=int)
        self.batch_years, self.batch_episode_lengths = [year], [self.length]
        self.envs = [CascadeReservoirEnv(self.data, self.physical_config,
                                       start_year=year, end_year=year, action_bins=self.action_bins)]
        self.episode_attempts += 1
        self._allocate_buffers()
        self.prep_rollout()
        obs = np.stack([env.actor_observations() for env in self.envs])
        state = np.stack([env.state() for env in self.envs])
        for i, b in enumerate(self.actor_buffer):
            b.step = 0; b.obs[0] = obs[:, i]
            b.masks.fill(1); b.active_masks.fill(1)
        cb = self.critic_buffer
        cb.step = 0; cb.share_obs[0] = state
        cb.masks.fill(1); cb.bad_masks.fill(1)
        energy, forced, largest_residual = np.zeros(self.n_envs), 0, 0.
        reward_metrics = EpisodeRewardMetrics(self.algo_args['algo']['gamma'])
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
            next_obs = np.stack([env.actor_observations() for env in self.envs])
            next_state = np.stack([env.state() for env in self.envs])
            controls = np.stack([r["controls"] for r in results])
            effective_actions = np.stack([r["actions"] for r in results])
            forced += int((controls == 0).sum())
            largest_residual = max(largest_residual, max(float(np.abs(r["residual"]).max()) for r in results))
            next_masks = (~done).astype(np.float32)[:, None]
            for i, b in enumerate(self.actor_buffer):
                lp = np.where(controls[:, i:i+1] > 0, log_probs[i], 0.)
                b.insert(next_obs[:, i], effective_actions[:, i:i+1], lp, next_masks,
                         np.ones((self.n_envs, 1), dtype=np.float32))
                # Controllability belongs to the sampled action at t, not t+1.
                b.active_masks[t] = controls[:, i:i+1]
            rewards = np.array([[r["reward"]] for r in results], dtype=np.float32)
            cb.insert(next_state, _t2n(values), rewards, next_masks,
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

    def collect_batch(self, episodes):
        """Collect complete independent years under one unchanged actor/critic.

        GAE is computed by collect_episode BEFORE concatenation. Terminal masks
        and episode-start time weights survive concatenation; no leap-day pads
        or artificial transitions are passed to the optimizer.
        """
        episodes = validate_rollout_episodes(episodes)
        actors, critics, rollouts, metrics = [], [], [], []
        years = []
        for _ in range(episodes):
            self.collect_episode()
            actors.append(self.actor_buffer)
            critics.append(self.critic_buffer)
            rollouts.append(dict(self.last_rollout))
            metrics.append(dict(self.last_reward_metrics))
            years.append(self.sampled_years[-1])
        lengths = [b.episode_length for b in critics]
        self.batch_years, self.batch_episode_lengths = years, lengths
        self.batch_episode_starts = np.r_[0, np.cumsum(lengths)[:-1]]
        self.length = sum(lengths)
        self.algo_args['train']['episode_length'] = self.length
        for key in ('actor_num_mini_batch', 'critic_num_mini_batch'):
            requested = int(self.requested_args['algo'][key])
            self.algo_args['algo'][key] = max(n for n in range(1, min(requested,self.length)+1)
                                             if self.length % n == 0)
        self._allocate_buffers()

        def state_rows(buffers, field):
            # At a join, the next year's initial state replaces the preceding
            # terminal state. Its value is ignored by the zero terminal mask.
            return np.concatenate([getattr(b,field)[:-1] for b in buffers]+
                                  [getattr(buffers[-1],field)[-1:]], axis=0)

        def boundary_rows(buffers, field):
            # Preserve each preceding transition's terminal mask, including at
            # the next year's first state, where the discount clock must reset.
            return np.concatenate([getattr(buffers[0],field)[:1]]+
                                  [getattr(b,field)[1:] for b in buffers], axis=0)

        for i, target in enumerate(self.actor_buffer):
            sources = [buffers[i] for buffers in actors]
            for field in ('obs','active_masks','available_actions'):
                if getattr(target,field) is not None:
                    getattr(target,field)[:] = state_rows(sources,field)
            target.masks[:] = boundary_rows(sources,'masks')
            for field in ('actions','action_log_probs'):
                getattr(target,field)[:] = np.concatenate([getattr(b,field) for b in sources],axis=0)
        target = self.critic_buffer
        for field in ('share_obs','value_preds','returns'):
            getattr(target,field)[:] = state_rows(critics,field)
        for field in ('masks','bad_masks'):
            getattr(target,field)[:] = boundary_rows(critics,field)
        target.rewards[:] = np.concatenate([b.rewards for b in critics],axis=0)
        # Episode diagnostics are batch MEANS, never a discounted reward chain
        # across unrelated years. This keeps their scale independent of K.
        self.last_reward_metrics = {key:(metrics[0][key] if key=='gamma' else
            float(np.mean([m[key] for m in metrics]))) for key in metrics[0]}
        self.last_rollout = dict(rollouts[-1])
        self.last_rollout.update(batch_episodes=episodes, batch_steps=self.length,
            episode_energy_TWh=float(np.mean([r['episode_energy_TWh'] for r in rollouts])),
            episode_mean_daily_energy_MWh=float(np.average(
                [r['episode_mean_daily_energy_MWh'] for r in rollouts],weights=lengths)),
            forced_fraction=float(np.average([r['forced_fraction'] for r in rollouts],weights=lengths)),
            max_balance_residual_m3=max(r['max_balance_residual_m3'] for r in rollouts))
        prediction = target.value_preds[:-1]
        if self.value_normalizer is not None:
            prediction = self.value_normalizer.denormalize(prediction)
        returns = target.returns[:-1]
        self.last_rollout.update(gae_target_mean=float(returns.mean()),
            value_prediction_mean=float(prediction.mean()),
            gae_target_rmse=float(np.sqrt(np.mean((returns-prediction)**2))),
            gae_target_explained_variance=float(1.-np.var(returns-prediction)/max(float(np.var(returns)),1e-8)))

    def evaluate_checkpoint(self):
        """Keep in-sample diagnostic checkpoints separate from held-out validation."""
        if self.fixed_training_year is not None:
            return self.evaluate_training_year()
        if self.validation_enabled:
            return self.validate()
        return None

    def _record_policy_diagnostics(self, rows, previous_checkpoint, scope):
        """Compare sampled and modal policies only at requested milestones.

        These reports observe learning; checkpoint selection still uses the
        original deterministic reward. Repeated samples never enter buffers.
        """
        if not self.diagnostics_enabled:
            return
        experiment = self.algo_args['experiment']
        milestones = experiment.get('diagnostic_eval_episodes', [0,20,100,150,400])
        final_episode = getattr(self, '_final_training_episode', experiment['episodes'])
        if not (any(previous_checkpoint < m <= self.iteration for m in milestones) or
                self.iteration == final_episode):
            return
        from harl.envs.cascade_reservoir.diagnostics import (
            policy_evaluation_report, validation_report)
        history = self.validation_diagnostics if scope == 'validation' else self.fixed_year_diagnostics
        previous = history[-1] if history else None
        if scope == 'validation':
            report = validation_report(self, rows, previous_report=previous)
        else:
            report = policy_evaluation_report(self, rows, previous_report=previous, scope=scope)
        history.append(report)
        log_episode(dict(report, record_type='policy_evaluation_diagnostics', seed=self.seed),
                    self.episode_log)

    def evaluate_training_year(self):
        """Evaluate the current actors on a fresh copy of the fixed training year.

        Evaluation uses categorical modes, never adds samples to the training
        buffers. Optional diagnostics sample this same training year separately
        at the requested milestones, without invoking held-out validation.
        """
        if self.fixed_training_year is None:
            raise RuntimeError('A fixed training year is required for this diagnostic')
        with torch.random.fork_rng(devices=[]):
            row = evaluate_policy(self.actor, self.data, self.physical_config,
                self.fixed_training_year, self.seed, self.iteration, self.training_steps,
                deterministic=True, diagnostics=True, gamma=self.algo_args['algo']['gamma'])
        power = float(row['episode_generation_reward'])
        terms = {term:term for term in PENALTY_TERMS}
        costs = {name: float(row.get('episode_penalty_'+term, 0.)) for name, term in terms.items()}
        total = float(row['episode_reward'])
        if (not np.isfinite([power, total]+list(costs.values())).all() or
                not np.isclose(total, power-sum(costs.values()), rtol=1e-9, atol=1e-9)):
            raise ValueError('Invalid fixed-year evaluation reward components')
        if row['feasible'] is not True or row['violation_count'] != 0:
            raise ValueError('Infeasible fixed-year evaluation')
        percentages = {term: 100.*costs[name]/power if power > 0. else None
                       for name, term in terms.items()}
        percentages['total'] = 100.*sum(costs.values())/power if power > 0. else None
        record = dict(record_type='fixed_year_evaluation', seed=self.seed, update=self.update_count,
            episode=self.iteration, training_steps=self.training_steps,
            year=self.fixed_training_year, deterministic=True,
            actual_generation_TWh=float(row['energy_TWh']), power_reward=power,
            total_reward=total, total_soft_penalty=sum(costs.values()),
            **{name+'_penalty': value for name, value in costs.items()},
            penalty_to_power_percent=percentages,
            gamma=float(self.algo_args['algo']['gamma']),
            reward_aggregation=REWARD_AGGREGATION,
            reward_definition=REWARD_DEFINITION,
            gae_lambda=float(self.algo_args['algo']['gae_lambda']),
            reward_penalties=dict(self.reward_settings),
            feasible=row['feasible'], violation_count=row['violation_count'],
            evaluation=row)
        if self.diagnostics_enabled and self.diagnostics_history:
            # These critic/actor statistics belong to the latest sampled
            # training rollout, separately from this deterministic evaluation.
            record['last_training_update_diagnostics'] = self.diagnostics_history[-1]
        self.fixed_year_history.append(record)
        score = total
        if score > self.best_score:
            self.best_score, self.best_energy, self.best_iteration = score, float(row['energy_MWh']), self.iteration
            self.best_training_steps, self.best_update = self.training_steps, self.update_count
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        log_episode(record, self.episode_log)
        print('seed={} episode={} fixed_training_year={} deterministic_generation={:.3f} TWh '
              'power={:.6f} guaranteed_output={:.6f} ecological_flow={:.6f} total={:.6f} '
              '(in-sample diagnostic)'.format(self.seed, self.iteration, self.fixed_training_year,
                  row['energy_TWh'], power, costs['guaranteed_output'], costs['ecological_flow'], total), flush=True)
        previous = self.fixed_year_history[-2]['episode'] if len(self.fixed_year_history)>1 else -1
        self._record_policy_diagnostics([row], previous, 'fixed_training_year')
        return score

    def validate(self):
        if not self.validation_enabled:
            raise ValueError('No validation years configured')
        rows = [evaluate_policy(self.actor, self.data, self.physical_config, year,
                                self.seed, self.iteration, self.training_steps,
                                gamma=self.algo_args['algo']['gamma'],
                                **({"diagnostics": True} if self.diagnostics_enabled else {}))
                for year in self.data.validation_years]
        energy = float(np.mean([row["energy_MWh"] for row in rows]))
        score = float(np.mean([row['episode_reward'] for row in rows]))
        if not np.isfinite(energy) or not np.isfinite(score):
            raise FloatingPointError('Non-finite validation score')
        history = dict(episode=self.iteration, update=self.update_count, training_steps=self.training_steps,
                       mean_TWh=energy/1e6, mean_reward=score,
                       mean_penalty=float(np.mean([row['episode_penalty'] for row in rows])),
                       gamma=self.algo_args['algo']['gamma'], years=rows)
        self.validation_history.append(history)
        if self.rollout_episodes > 1:
            log_episode(dict(record_type='validation_checkpoint',seed=self.seed,
                             rollout_episodes_per_update=self.rollout_episodes,**history),self.episode_log)
        if score > self.best_score:
            self.best_score, self.best_energy, self.best_iteration = score, energy, self.iteration
            self.best_training_steps, self.best_update = self.training_steps, self.update_count
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        year_label = ','.join(str(year) for year in self.data.validation_years)
        if self.reward_settings['enabled']:
            print('seed={} episode={} steps={} validation_years={} mean={:.3f} TWh reward={:.6f} '
                  'penalty={:.6f} best_reward={:.6f} selected_generation={:.3f} TWh'.format(
                self.seed,self.iteration,self.training_steps,year_label,energy/1e6,score,history['mean_penalty'],
                self.best_score,self.best_energy/1e6),flush=True)
        else:
            print("seed={} episode={} steps={} validation_years={} mean={:.3f} TWh "
                  "reward={:.6f} best_reward={:.6f} selected_generation={:.3f} TWh".format(
                self.seed, self.iteration, self.training_steps, year_label, energy / 1e6,
                score, self.best_score, self.best_energy / 1e6), flush=True)
        previous = self.validation_history[-2]['episode'] if len(self.validation_history)>1 else -1
        self._record_policy_diagnostics(rows, previous, 'validation')
        return score

    def run(self, episodes, eval_interval):
        if episodes < 1 or eval_interval < 1:
            raise ValueError("Positive training episodes/evaluation interval required")
        self._final_training_episode = episodes
        self.evaluate_checkpoint()  # Optional evaluation; training curves start after collection.
        updates = (episodes+self.rollout_episodes-1)//self.rollout_episodes
        for k, start in enumerate(range(0, episodes, self.rollout_episodes),1):
            count = min(self.rollout_episodes, episodes-start)
            self.iteration = start+count  # Always counts annual episodes, not updates.
            if self.algo_args["train"]["use_linear_lr_decay"]:
                for actor in self.actor:
                    actor.lr_decay(k, updates)
                self.critic.lr_decay(k, updates)
            if self.rollout_episodes == 1:
                self.collect_episode()
                self.batch_episode_starts = np.array([0],dtype=int)
                self.batch_years = self.sampled_years[-1:]
                self.batch_episode_lengths = [self.length]
            else:
                self.collect_batch(count)
            diagnostic_before = None
            if self.diagnostics_enabled:
                from harl.envs.cascade_reservoir.diagnostics import before_update
                diagnostic_before = before_update(self)
            self.prep_training()
            actor_info, critic_info = super().train()  # Original sequential HAPPO implementation.
            self.update_count += 1
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
            if self.rollout_episodes > 1 or self.joint_kl_info is not None or not self.validation_enabled:
                log_episode(dict(record_type='training_update',seed=self.seed,update=self.update_count,
                    episode=self.iteration,training_steps=self.training_steps,
                    years=self.batch_years,episode_lengths=self.batch_episode_lengths,
                    ppo_epochs_max=self.requested_args['algo']['ppo_epoch'],
                    actor_target_kl=self.requested_args['algo'].get('target_kl'),
                    actor_kl_rollback=self.requested_args['algo'].get('kl_rollback', False),
                    joint_target_kl=self.requested_args['algo'].get('joint_target_kl'),
                    joint_kl=self.joint_kl_info,
                    actor_optimization=[{k:float(info[k]) for k in (
                        'optimizer_steps','optimizer_attempts','optimizer_rejected_steps',
                        'ppo_epochs_completed','ppo_epochs_attempted','kl_early_stopped',
                        'policy_kl','rejected_policy_kl','joint_target_kl','joint_policy_kl',
                        'joint_actor_kl','joint_kl_remaining','rejected_joint_policy_kl',
                        'joint_kl_early_stopped','joint_kl_before','joint_actor_budget') if k in info}
                        for info in actor_info],
                    **(dict(episode_reward_mean=self.last_reward_metrics['episode_reward'],
                            episode_generation_TWh_mean=self.last_rollout['episode_energy_TWh'],
                            episode_horizon_days=self.length, parallel_envs=self.n_envs)
                       if self.collection_mode == 'five_year_parallel' else
                       dict(annual_reward_mean=self.last_reward_metrics['episode_reward'],
                            annual_generation_TWh_mean=self.last_rollout['episode_energy_TWh']))),self.episode_log)
            if self.diagnostics_enabled:
                from harl.envs.cascade_reservoir.diagnostics import after_update, print_update_window
                report = after_update(self, diagnostic_before)
                if self.joint_kl_info is not None:
                    report['joint_kl'] = self.joint_kl_info
                for actor_row, info in zip(report['actors'], actor_info):
                    actor_row['optimization'] = {key: float(info[key]) for key in (
                        'optimizer_steps','optimizer_attempts','optimizer_rejected_steps',
                        'ppo_epochs_completed','ppo_epochs_attempted','kl_early_stopped',
                        'policy_kl','rejected_policy_kl','joint_target_kl','joint_policy_kl',
                        'joint_actor_kl','joint_kl_remaining','rejected_joint_policy_kl',
                        'joint_kl_early_stopped','joint_kl_before','joint_actor_budget') if key in info}
                    actor_row['surrogate'] = {key: float(value) for key, value in info.items()
                        if key.startswith(('clipped_surrogate_', 'unclipped_surrogate_',
                                           'weighted_entropy_', 'regularized_objective_'))}
                self.diagnostics_history.append(report)
                log_episode(dict(report, record_type='training_diagnostics', seed=self.seed),
                            self.episode_log)
                if k % eval_interval == 0 or k == updates:
                    print_update_window(self, min(eval_interval,k))
            if k % eval_interval == 0 or k == updates:
                self.evaluate_checkpoint()
        if self.fixed_training_year is not None:
            result = dict(seed=self.seed, policy_selection='fixed_training_year',
                          fixed_training_year=self.fixed_training_year,
                          updates_completed=self.update_count,
                          best_episode=self.best_iteration, best_training_year_TWh=self.best_energy/1e6,
                          best_training_steps=self.best_training_steps, best_update=self.best_update,
                          best_training_year_reward=self.best_score)
            return result
        if not self.validation_enabled:
            return dict(seed=self.seed, updates_completed=self.update_count,
                        **self.test_policy_metadata())
        result = dict(seed=self.seed, policy_selection='validation',
                      best_episode=self.best_iteration, best_iteration=self.best_iteration,
                      best_training_steps=self.best_training_steps, best_update=self.best_update,
                      updates_completed=self.update_count,
                      best_validation_TWh=self.best_energy / 1e6, best_validation_reward=self.best_score)
        return result

    def test_policy_metadata(self):
        """Identify the final trained policy or an explicitly validation-selected one."""
        if self.fixed_training_year is not None:
            raise RuntimeError('Fixed-training-year diagnostics do not run held-out tests')
        if not self.validation_enabled:
            if self.update_count < 1:
                raise RuntimeError('No trained final policy is available')
            return dict(policy_selection='final_training_update', selected_episode=self.iteration,
                        selected_training_steps=self.training_steps, selected_update=self.update_count)
        if self.best_actor_states is None:
            raise RuntimeError('No validation-selected policy is available')
        if self.best_training_steps is None or self.best_update is None:
            raise RuntimeError('Validation-selected checkpoint training metadata is missing')
        return dict(policy_selection='validation', selected_episode=self.best_iteration,
                    selected_training_steps=self.best_training_steps, selected_update=self.best_update)

    def final_test(self):
        """Test the final actors, or the best actors when validation was requested."""
        selection = self.test_policy_metadata()
        if self.validation_enabled:
            for actor, weights in zip(self.actor, self.best_actor_states):
                actor.actor.load_state_dict(weights)
        return [dict(evaluate_policy(self.actor, self.data, self.physical_config, year,
                                    self.seed, selection['selected_episode'], selection['selected_training_steps'],
                                    gamma=self.algo_args['algo']['gamma']),
                     typical_label=label, **selection,
                     run_training_steps=self.training_steps)
                for year, label in zip(self.data.test_years, self.data.test_labels)]
