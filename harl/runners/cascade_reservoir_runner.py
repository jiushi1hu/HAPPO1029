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
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.utils.trans_tools import _t2n


def configure_calendar(algo_args, data, year=None):
    """Allocate a complete sampled year without padding or dropping leap days."""
    args = copy.deepcopy(algo_args)
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

    def __init__(self, data, physical_config, algo_args, seed):
        if algo_args["model"].get("action_distribution") != "beta":
            raise ValueError("Cascade normalized controls require the bounded Beta action distribution")
        for key in ("use_policy_active_masks", "mask_inactive_factors"):
            if not algo_args["algo"].get(key, False):
                raise ValueError("Forced cascade actions require " + key)
        self.data, self.physical_config = data, physical_config
        self.requested_args = copy.deepcopy(algo_args)
        self.algo_args = algo_args = configure_calendar(algo_args, data)
        self.training_years = data.training_years
        self.year_rng = np.random.RandomState(int(seed))
        self.sampled_years = []
        self.seed = int(seed)
        self.num_agents, self.state_type = 5, "EP"
        self.share_param = False
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.action_aggregation = "prod"
        self.device = torch.device("cpu")
        self.value_normalizer = ValueNorm(1, device=self.device) if algo_args["train"]["use_valuenorm"] else None
        torch.set_num_threads(algo_args["device"]["torch_threads"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        self.recurrent_n = 1
        self.rnn_hidden_size = algo_args["model"]["hidden_sizes"][-1]
        self.n_envs = algo_args["train"]["n_rollout_threads"]
        self.length = algo_args["train"]["episode_length"]
        if algo_args["algo"]["gamma"] != 1.:
            raise ValueError("The undiscounted generation objective requires gamma=1")
        if algo_args["model"]["use_recurrent_policy"] or algo_args["model"]["use_naive_recurrent_policy"]:
            raise ValueError("The cascade adapter currently supports feed-forward actors only")
        if algo_args["algo"]["share_param"]:
            raise ValueError("The cascade specification requires independent actors")
        for key in ("actor_num_mini_batch", "critic_num_mini_batch"):
            if self.length * self.n_envs % algo_args["algo"][key]:
                raise ValueError("Choose minibatch count dividing rollout size; avoid dropping samples")
        self.envs = [CascadeReservoirEnv(data, physical_config, start_year=self.training_years[0],
                                       end_year=self.training_years[0])]
        env = self.envs[0]
        network_args = {**algo_args["model"], **algo_args["algo"]}
        self.actor = [HAPPO(network_args, env.observation_space[i], env.action_space[i], self.device) for i in range(5)]
        self.critic = VCritic(network_args, env.share_observation_space[0], self.device)
        self._allocate_buffers()
        self.iteration, self.training_steps = 0, 0
        self.best_energy = -float("inf")
        self.best_iteration = None
        self.started = time.monotonic()
        self.last_rollout = {}
        self.completed_episodes = 0
        self.validation_history = []
        self.training_history = []
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
        return int(self.year_rng.choice(self.training_years))

    @torch.no_grad()
    def collect_episode(self, year=None):
        """Sample one year with replacement; reset storage and collect every day."""
        year = self.sample_training_year() if year is None else int(year)
        self.algo_args = configure_calendar(self.requested_args, self.data, year)
        self.length = self.algo_args["train"]["episode_length"]
        self.envs = [CascadeReservoirEnv(self.data, self.physical_config,
                                       start_year=year, end_year=year)]
        self._allocate_buffers()
        self.prep_rollout()
        obs = np.stack([env.observation() for env in self.envs])
        for b in self.actor_buffer:
            b.step = 0; b.obs[0] = obs
            b.masks.fill(1); b.active_masks.fill(1); b.rnn_states.fill(0)
        cb = self.critic_buffer
        cb.step = 0; cb.share_obs[0] = obs
        cb.masks.fill(1); cb.bad_masks.fill(1); cb.rnn_states_critic.fill(0)
        energy, forced, largest_residual = np.zeros(self.n_envs), 0, 0.
        for t in range(self.length):
            actions, log_probs, rnn = [], [], []
            for i in range(5):
                b = self.actor_buffer[i]
                a, lp, h = self.actor[i].get_actions(b.obs[t], b.rnn_states[t], b.masks[t])
                actions.append(_t2n(a)); log_probs.append(_t2n(lp)); rnn.append(_t2n(h))
            actions = np.stack(actions, axis=1)
            values, hidden = self.critic.get_values(cb.share_obs[t], cb.rnn_states_critic[t], cb.masks[t])
            for e, env in enumerate(self.envs):
                env.step(actions[e])
                energy[e] += env.last_transition["energy"].sum()
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
                b.insert(next_obs, rnn[i], effective_actions[:, i:i+1], lp, next_masks,
                         np.ones((self.n_envs, 1), dtype=np.float32))
                # Controllability belongs to the sampled action at t, not t+1.
                b.active_masks[t] = controls[:, i:i+1]
            rewards = np.array([[r["reward"]] for r in results], dtype=np.float32)
            cb.insert(next_obs, _t2n(hidden), _t2n(values), rewards, next_masks,
                      np.ones_like(next_masks))
        self.completed_episodes += 1
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

    def validate(self):
        rows = [evaluate_policy(self.actor, self.data, self.physical_config, year,
                                self.seed, self.iteration, self.training_steps,
                                **({"diagnostics": True} if self.diagnostics_enabled else {}))
                for year in self.data.validation_years]
        score = float(np.mean([row["energy_MWh"] for row in rows]))
        self.validation_history.append(dict(episode=self.iteration,
            training_steps=self.training_steps, mean_TWh=score / 1e6, years=rows))
        if score > self.best_energy:
            self.best_energy, self.best_iteration = score, self.iteration
            self.best_actor_states = [copy.deepcopy(a.actor.state_dict()) for a in self.actor]
        print("seed={} episode={} steps={} validation_2020_2024_mean={:.3f} TWh best={:.3f} TWh".format(
            self.seed, self.iteration, self.training_steps, score / 1e6, self.best_energy / 1e6), flush=True)
        if self.diagnostics_enabled:
            milestones = self.algo_args["experiment"].get("diagnostic_eval_episodes", [0,20,100,150,400])
            if self.iteration in milestones or self.iteration == self.algo_args["experiment"]["episodes"]:
                from harl.envs.cascade_reservoir.diagnostics import validation_report
                self.validation_diagnostics.append(validation_report(self, rows))
        return score

    def run(self, episodes, eval_interval):
        if episodes < 1 or eval_interval < 1:
            raise ValueError("Positive training episodes/evaluation interval required")
        self.validate()  # Honest untrained point on the learning curve.
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
                       elapsed_seconds=time.monotonic() - self.started, **self.last_rollout)
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
                self.validate()
        return dict(seed=self.seed, best_episode=self.best_iteration, best_iteration=self.best_iteration,
                    best_validation_TWh=self.best_energy / 1e6)

    def final_test(self):
        """Evaluate selected actors once on each user-defined typical year."""
        if self.best_actor_states is None:
            raise RuntimeError("No validation-selected policy is available")
        for actor, weights in zip(self.actor, self.best_actor_states):
            actor.actor.load_state_dict(weights)
        return [dict(evaluate_policy(self.actor, self.data, self.physical_config, year,
                                    self.seed, self.best_iteration, self.training_steps),
                     typical_label=label)
                for year, label in zip(self.data.test_years, self.data.test_labels)]
