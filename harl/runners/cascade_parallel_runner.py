"""Fixed-horizon cascade HAPPO with independent process-parallel collectors."""
import copy
import numpy as np
import torch
from harl.envs.cascade_reservoir.episodes import (
    episode_environment, positive_integer, training_windows)
from harl.envs.cascade_reservoir.parallel_env import ParallelCascadeEnvs
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.training_metrics import log_episode
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.utils.trans_tools import _t2n


class CascadeParallelRunner(CascadeReservoirRunner):
    collection_mode = 'five_year_parallel'

    def __init__(self, data, physical_config, algo_args, seed, episode_log=None):
        days = positive_integer(algo_args['train']['episode_length'], 'episode_days')
        count = positive_integer(algo_args['train']['n_rollout_threads'], 'parallel_envs')
        if algo_args['experiment'].get('fixed_training_year') is not None:
            raise ValueError('Fixed-year diagnostics require annual collection mode')
        if algo_args['experiment']['rollout_episodes'] != count:
            raise ValueError('Parallel rollout_episodes must equal the number of environments')
        self.windows = training_windows(data, days)
        # Reuse actor/critic construction, not the legacy annual collector.
        legacy = copy.deepcopy(algo_args)
        legacy['train']['n_rollout_threads'] = 1
        super().__init__(data, physical_config, legacy, seed, episode_log)
        self.requested_args = copy.deepcopy(algo_args)
        self.algo_args = copy.deepcopy(algo_args)
        self.length, self.n_envs = days, count
        self.training_years = tuple(self.windows)
        self.envs = [episode_environment(data, physical_config,
            self.windows[self.training_years[0]], days, self.action_bins)]
        self._allocate_buffers()
        self.batch_episode_starts = np.zeros(count, dtype=int)
        self.parallel_envs = None
        self.worker_pids = []

    def close(self):
        if self.parallel_envs is not None:
            self.parallel_envs.close()
            self.parallel_envs = None

    def run(self, episodes, eval_interval):
        episodes = positive_integer(episodes, 'episodes')
        if episodes % self.n_envs:
            raise ValueError('Total episodes must be divisible by parallel_envs; no padded or dropped lanes')
        try:
            return super().run(episodes, eval_interval)
        finally:
            self.close()

    def collect_episode(self, year=None):
        if self.n_envs != 1:
            raise ValueError('Use collect_batch to collect all parallel lanes together')
        return self.collect_batch(1)

    @torch.no_grad()
    def collect_batch(self, episodes):
        if episodes != self.n_envs:
            raise ValueError('A collection round must contain one full episode per parallel environment')
        years = [self.sample_training_year() for _ in range(self.n_envs)]
        if any(year not in self.windows for year in years):
            raise ValueError('Sampled window includes unavailable or held-out daily data')
        self.batch_years = years
        self.batch_episode_lengths = [self.length]*self.n_envs
        self.batch_episode_starts = np.zeros(self.n_envs, dtype=int)
        if self.parallel_envs is None:
            self.parallel_envs = ParallelCascadeEnvs(self.data, self.physical_config,
                self.action_bins, self.length, self.n_envs, self.algo_args['algo']['gamma'],
                self.algo_args['algo']['gae_lambda'], self.seed)
        ids = list(range(self.episode_attempts+1, self.episode_attempts+self.n_envs+1))
        self.episode_attempts += self.n_envs
        self._allocate_buffers()
        self.prep_rollout()
        try:
            initial = self.parallel_envs.reset([self.windows[year] for year in years], ids)
            self.worker_pids = [r['pid'] for r in initial]
            obs = np.stack([r['obs'] for r in initial])
            state = np.stack([r['state'] for r in initial])
            for i, b in enumerate(self.actor_buffer):
                b.obs[0] = obs[:, i]
            cb = self.critic_buffer
            cb.share_obs[0] = state
            forced, largest_residual = 0, 0.
            for t in range(self.length):
                actions, log_probs = [], []
                for i, b in enumerate(self.actor_buffer):
                    action, log_prob = self.actor[i].get_actions(b.obs[t])
                    actions.append(_t2n(action)); log_probs.append(_t2n(log_prob))
                values = self.critic.get_values(cb.share_obs[t])
                results = self.parallel_envs.step(np.stack(actions, axis=1))
                transitions = [r['transition'] for r in results]
                done = np.array([r['done'] for r in results])
                if not np.all(done == (t == self.length-1)):
                    raise RuntimeError('Unexpected early termination or incomplete parallel episode')
                obs = np.stack([r['obs'] for r in results])
                state = np.stack([r['state'] for r in results])
                controls = np.stack([r['controls'] for r in transitions])
                effective_actions = np.stack([r['actions'] for r in transitions])
                next_masks = (~done).astype(np.float32)[:, None]
                for i, b in enumerate(self.actor_buffer):
                    lp = np.where(controls[:, i:i+1] > 0, log_probs[i], 0.)
                    b.insert(obs[:, i], effective_actions[:, i:i+1], lp, next_masks,
                             np.ones((self.n_envs, 1), dtype=np.float32))
                    b.active_masks[t] = controls[:, i:i+1]
                cb.insert(state, _t2n(values),
                    np.array([[r['reward']] for r in transitions], dtype=np.float32),
                    next_masks, np.ones_like(next_masks))
                forced += int((controls == 0).sum())
                largest_residual = max(largest_residual,
                    max(float(np.abs(r['residual']).max()) for r in transitions))
        except Exception as exc:
            for record in getattr(exc, 'episode_records', []):
                self.hard_failed_episodes += 1
                record.update(failed_episodes_so_far=self.hard_failed_episodes,
                    hard_constraint_failure_rate_so_far=self.hard_failed_episodes/self.episode_attempts)
                log_episode(record, self.episode_log)
            self.close()
            raise
        metrics = []
        for lane, result in enumerate(results):
            record = result['episode_record']
            self.collected_training_steps += self.length
            self.completed_episodes += 1
            record.update(parallel_env_index=lane, parallel_envs=self.n_envs,
                rollout_episodes_per_update=self.n_envs, collecting_for_update=self.update_count+1,
                policy_updates_completed=self.update_count, training_steps=self.collected_training_steps,
                ppo_epochs_max=self.requested_args['algo']['ppo_epoch'],
                actor_target_kl=self.requested_args['algo'].get('target_kl'),
                actor_kl_rollback=self.requested_args['algo'].get('kl_rollback', False),
                joint_target_kl=self.requested_args['algo'].get('joint_target_kl'))
            self.episode_history.append(record)
            log_episode(record, self.episode_log)
            metrics.append(result['reward_metrics'])
        self.sampled_years.extend(years)
        self.last_reward_metrics = {
            key: (self.algo_args['algo']['gamma'] if key == 'gamma' else
                  float(np.mean([m[key] for m in metrics]))) for key in metrics[0]}
        self.last_rollout = dict(sampled_year=years[0], episode_days=self.length,
            episode_energy_TWh=float(np.mean([r['episode_record']['actual_generation_TWh'] for r in results])),
            episode_mean_daily_energy_MWh=float(np.mean([
                r['episode_record']['actual_generation_MWh'] for r in results])/self.length),
            completed_episodes=self.completed_episodes, batch_episodes=self.n_envs,
            batch_total_days=self.length*self.n_envs, batch_unique_years=len(set(years)),
            forced_fraction=forced/(self.length*self.n_envs*5), max_balance_residual_m3=largest_residual)
        self.compute()
        predictions = cb.value_preds[:-1]
        if self.value_normalizer is not None:
            predictions = self.value_normalizer.denormalize(predictions)
        targets = cb.returns[:-1]
        self.last_rollout.update(gae_target_mean=float(targets.mean()),
            value_prediction_mean=float(predictions.mean()),
            gae_target_rmse=float(np.sqrt(np.mean((targets-predictions)**2))),
            gae_target_explained_variance=float(1.-np.var(targets-predictions)/max(float(np.var(targets)),1e-8)))

    def _evaluate_sequence(self, years, episode, steps):
        modes = [a.actor.training for a in self.actor]
        try:
            with torch.random.fork_rng(devices=[]):
                return evaluate_policy(self.actor, self.data, self.physical_config, years[0],
                    self.seed, episode, steps, diagnostics=True,
                    gamma=self.algo_args['algo']['gamma'], source_years=years, episode_days=self.length)
        finally:
            for actor, mode in zip(self.actor, modes):
                actor.actor.train(mode)

    def validate(self):
        if not self.validation_enabled:
            raise ValueError('No validation years configured')
        row = self._evaluate_sequence(list(self.data.validation_years), self.iteration, self.training_steps)
        score = row['episode_reward']
        if score > self.best_score:
            self.best_score, self.best_energy = score, row['energy_MWh']
            self.best_iteration, self.best_training_steps = self.iteration, self.training_steps
            self.best_update = self.update_count
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        history = dict(seed=self.seed, episode=self.iteration, update=self.update_count,
            training_steps=self.training_steps, mean_reward=score, mean_TWh=row['energy_TWh'],
            mean_penalty=row['episode_penalty'], gamma=self.algo_args['algo']['gamma'],
            episode_horizon_days=self.length, years=[row])
        self.validation_history.append(history)
        log_episode(dict(history, record_type='validation_checkpoint'), self.episode_log)
        return score

    def final_test(self):
        selection = self.test_policy_metadata()
        if self.validation_enabled:
            for actor, weights in zip(self.actor, self.best_actor_states):
                actor.actor.load_state_dict(weights)
        row = self._evaluate_sequence(list(self.data.test_years), selection['selected_episode'],
                                      selection['selected_training_steps'])
        row.update(record_type='final_test_sequence', **selection,
                   run_training_steps=self.training_steps, typical_label='representative five-year sequence')
        log_episode(row, self.episode_log)
        return [row]
