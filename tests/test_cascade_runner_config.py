"""Runner options must have the same execution semantics as the HARL runner."""
import copy
import io
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from examples.train_cascade import main
from cascade_test_utils import load_configuration
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner, validate_seed
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

    def test_gamma_is_validated_for_direct_runner_configuration(self):
        for gamma in (-.01, 1.01, np.nan, np.inf, -np.inf, None, True, 'invalid'):
            with self.subTest(gamma=gamma):
                cfg = copy.deepcopy(self.cfg)
                cfg['algo']['gamma'] = gamma
                with self.assertRaisesRegex(ValueError, 'gamma must be finite and between 0 and 1'):
                    CascadeReservoirRunner(self.data, self.physical, cfg, 5)

    def test_cli_gamma_overrides_yaml_and_is_reported(self):
        for override in (None, 0., .99, 1.):
            with self.subTest(override=override):
                cfg = copy.deepcopy(self.cfg)
                cfg['algo']['gamma'] = .9
                argv = ['train_cascade.py', '--check-data-only']
                if override is not None:
                    argv += ['--gamma', str(override)]
                output = io.StringIO()
                with patch('sys.argv', argv), \
                     patch('examples.train_cascade.load_configuration', return_value=(cfg, self.physical)), \
                     patch('examples.train_cascade.CascadeData', return_value=self.data), \
                     redirect_stdout(output):
                    main()
                startup, _ = json.JSONDecoder().raw_decode(output.getvalue())
                self.assertEqual(startup['gamma'], .9 if override is None else override)
                self.assertEqual(startup['evaluation_metric'], 'undiscounted annual generation')

    def test_cli_rejects_invalid_gamma_before_loading_data(self):
        for value in ('-.01', '1.01', 'nan', 'inf', '-inf'):
            with self.subTest(value=value), patch('sys.argv', [
                    'train_cascade.py', '--check-data-only', '--gamma='+value]), \
                 patch('examples.train_cascade.CascadeData') as load, \
                 redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as failure:
                    main()
                self.assertEqual(failure.exception.code, 2)
                self.assertIn('gamma must be finite and between 0 and 1', error.getvalue())
                load.assert_not_called()

    def assert_early_cli_rejection(self, cfg, options, message):
        error=io.StringIO()
        with patch('sys.argv',['train_cascade.py']+options), \
             patch('examples.train_cascade.load_configuration',return_value=(cfg,dict(self.physical))), \
             patch('examples.train_cascade.CascadeData') as data, \
             patch('examples.train_cascade.validate_operating_feasibility') as feasibility, \
             patch('examples.train_cascade.reported_references') as references, \
             patch('examples.train_cascade.CascadeReservoirRunner') as runner,redirect_stderr(error):
            with self.assertRaises(SystemExit) as failure:main()
        self.assertEqual(failure.exception.code,2)
        self.assertIn(message,error.getvalue())
        for work in (data,feasibility,references,runner):work.assert_not_called()

    def test_yaml_and_direct_runner_reject_invalid_gae_lambda(self):
        for trace in (-.01,1.01,np.nan,np.inf,-np.inf,None,True,np.bool_(False),'invalid',[],10**400):
            with self.subTest(trace=trace):
                cfg=copy.deepcopy(self.cfg);cfg['algo']['gae_lambda']=trace
                self.assert_early_cli_rejection(cfg,['--check-data-only'],
                    'GAE lambda must be finite and between 0 and 1')
                with patch('harl.runners.cascade_reservoir_runner.CascadeReservoirEnv') as env, \
                     patch('harl.runners.cascade_reservoir_runner.np.random.RandomState') as rng:
                    with self.assertRaisesRegex(ValueError,'GAE lambda must be finite and between 0 and 1'):
                        CascadeReservoirRunner(self.data,self.physical,cfg,5)
                    env.assert_not_called();rng.assert_not_called()

    def test_cli_rejects_invalid_gae_lambda_before_loading_data(self):
        for value in ('-.01','1.01','nan','inf','-inf'):
            with self.subTest(value=value):
                self.assert_early_cli_rejection(copy.deepcopy(self.cfg),
                    ['--check-data-only','--gae-lambda='+value],
                    'GAE lambda must be finite and between 0 and 1')

    def test_valid_gae_lambda_values_and_overrides_reach_startup_and_buffers(self):
        for override in (None,0.,.95,1.):
            with self.subTest(override=override):
                cfg=copy.deepcopy(self.cfg)
                # A valid override must replace an invalid YAML value.
                cfg['algo']['gae_lambda']=.9 if override is None else np.nan
                argv=['train_cascade.py','--check-data-only']
                if override is not None:argv+=['--gae-lambda',str(override)]
                output=io.StringIO()
                with patch('sys.argv',argv), \
                     patch('examples.train_cascade.load_configuration',return_value=(cfg,dict(self.physical))), \
                     patch('examples.train_cascade.CascadeData',return_value=self.data), \
                     patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]), \
                     redirect_stdout(output):
                    main()
                expected=.9 if override is None else override
                startup,_=json.JSONDecoder().raw_decode(output.getvalue())
                self.assertEqual(startup['gae_lambda'],expected)
                runner=CascadeReservoirRunner(self.data,self.physical,cfg,5)
                self.assertEqual(runner.critic_buffer.gae_lambda,expected)

    def test_yaml_and_direct_runner_reject_invalid_seeds(self):
        invalid=(-1,2**32,1.5,1.,np.nan,np.inf,None,True,np.bool_(False),'1',[])
        for seed in invalid:
            with self.subTest(seed=seed):
                cfg=copy.deepcopy(self.cfg);cfg['experiment']['seeds']=[seed]
                self.assert_early_cli_rejection(cfg,['--check-data-only'],
                    'Seed must be an integer between 0 and 4294967295')
                with patch('harl.runners.cascade_reservoir_runner.CascadeReservoirEnv') as env, \
                     patch('harl.runners.cascade_reservoir_runner.np.random.RandomState') as rng:
                    with self.assertRaisesRegex(ValueError,'Seed must be an integer between 0 and 4294967295'):
                        CascadeReservoirRunner(self.data,self.physical,self.cfg,seed)
                    env.assert_not_called();rng.assert_not_called()

    def test_cli_rejects_out_of_range_seeds_before_any_data_work(self):
        for value in ('-1','4294967296'):
            with self.subTest(value=value):
                self.assert_early_cli_rejection(copy.deepcopy(self.cfg),['--seeds',value],
                    'Seed must be an integer between 0 and 4294967295')

    def test_seed_boundaries_numpy_integers_and_cli_override_are_supported(self):
        for seed in (0,2**32-1,np.int64(7),np.uint32(2**32-1)):
            with self.subTest(seed=seed):
                self.assertIs(type(validate_seed(seed)),int)
                runner=CascadeReservoirRunner(self.data,self.physical,self.cfg,seed)
                self.assertEqual(runner.seed,int(seed))
                self.assertEqual(runner.sample_training_year(),
                    int(np.random.RandomState(int(seed)).choice(self.data.training_years)))
        for override in (False,True):
            cfg=copy.deepcopy(self.cfg)
            cfg['experiment']['seeds']=[-1] if override else [0,np.uint32(2**32-1)]
            argv=['train_cascade.py','--check-data-only']
            if override:argv+=['--seeds','0','4294967295']
            output=io.StringIO()
            with patch('sys.argv',argv), \
                 patch('examples.train_cascade.load_configuration',return_value=(cfg,dict(self.physical))), \
                 patch('examples.train_cascade.CascadeData',return_value=self.data), \
                 patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]), \
                 redirect_stdout(output):
                main()
            self.assertEqual(cfg['experiment']['seeds'],[0,2**32-1])
            self.assertIn('Data checks passed',output.getvalue())

    def test_seed_sequence_must_still_be_nonempty_and_unique(self):
        for seeds in (None,1,'1',[],[1,1],[1,np.int64(1)]):
            with self.subTest(seeds=seeds):
                cfg=copy.deepcopy(self.cfg);cfg['experiment']['seeds']=seeds
                message=('Seeds must be a nonempty sequence' if not isinstance(seeds,list) else
                         'nonempty unique seeds are required')
                self.assert_early_cli_rejection(cfg,['--check-data-only'],message)

    def test_default_seasonality_and_cli_overrides_are_reported(self):
        from examples.train_cascade import load_configuration as defaults
        _,default_physical=defaults()
        self.assertIs(default_physical['seasonal_constraints'],True)
        for flag,enabled in ((None,True),('--no-seasonal-constraints',False),('--seasonal-constraints',True)):
            with self.subTest(flag=flag):
                argv=['train_cascade.py','--check-data-only']+([] if flag is None else [flag])
                output=io.StringIO()
                with patch('sys.argv',argv), \
                     patch('examples.train_cascade.load_configuration',return_value=(copy.deepcopy(self.cfg),dict(default_physical))), \
                     patch('examples.train_cascade.CascadeData',return_value=self.data), \
                     patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]) as check, \
                     patch('examples.train_cascade.reported_references') as reference,redirect_stdout(output):
                    main()
                startup,_=json.JSONDecoder().raw_decode(output.getvalue())
                self.assertEqual(startup['seasonal_constraints'],enabled)
                self.assertEqual(startup['observation_size'],43 if enabled else 18)
                self.assertEqual(startup['reference_type'],'seasonal_target_tracking' if enabled else 'constant_storage')
                self.assertEqual(check.call_args[0][1]['seasonal_constraints'],enabled)
                reference.assert_not_called()
                if enabled:
                    self.assertTrue(all(v=='scenario_reference_not_official_operating_rule'
                        for v in startup['operation_data_status'].values()))

    def test_reference_workflow_respects_test_holdout_and_selected_grid(self):
        for skip in (False,True):
            events=[];cfg=copy.deepcopy(self.cfg)
            cfg['experiment'].update(episodes=1,eval_interval=1,seeds=[1,2])
            class FakeRunner:
                def __init__(runner,data,physical,config,seed):
                    runner.seed=seed;runner.validation_history=[]
                def run(runner,*args):events.append(('run',runner.seed))
                def final_test(runner):
                    events.append(('test',runner.seed))
                    return [dict(year=y,typical_label=label,seed=runner.seed,energy_TWh=1.)
                        for y,label in zip(self.data.test_years,self.data.test_labels)]
            def references(data,physics,years,action_bins,phase):
                events.append((phase,tuple(years),action_bins))
                return [dict(year=y,energy_TWh=2.) for y in years]
            argv=['train_cascade.py','--no-plot','--action-bins','37']+(['--skip-final-test'] if skip else [])
            with patch('sys.argv',argv), \
                 patch('examples.train_cascade.load_configuration',return_value=(cfg,dict(self.physical))), \
                 patch('examples.train_cascade.CascadeData',return_value=self.data), \
                 patch('examples.train_cascade.validate_operating_feasibility',return_value=[1965]), \
                 patch('examples.train_cascade.CascadeReservoirRunner',FakeRunner), \
                 patch('examples.train_cascade.reported_references',side_effect=references), \
                 redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(events[:3],[('validation',self.data.validation_years,37),('run',1),('run',2)])
            self.assertEqual(events[3:],[] if skip else
                [('test',1),('test',2),('test',self.data.test_years,37)])

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
