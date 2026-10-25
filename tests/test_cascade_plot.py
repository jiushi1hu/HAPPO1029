"""In-memory plotting: episode alignment and no exported figure files."""
import tempfile
from pathlib import Path
import unittest
import io
import json
import copy
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples.plot_cascade import (make_figure, make_validation_reward_figure, make_training_figures,
                                  make_reward_scale_figures, make_fixed_year_figure, main)
import numpy as np


class PlotTests(unittest.TestCase):
    def fixed_year_payload(self):
        return dict(fixed_training_year=2013, seeds=[1, 2], fixed_year_histories=[[
            dict(seed=seed, episode=episode, year=2013, deterministic=True,
                 feasible=True, violation_count=0, actual_generation_TWh=power*seed,
                 power_reward=power*seed, band_penalty=seed, rise_penalty=0., fall_penalty=0.,
                 total_reward=(power-1)*seed)
            for episode, power in [(0, 10.), (5, 14.), (10, 11.)]] for seed in (1, 2)])

    def test_fixed_year_plot_uses_unsmoothed_deterministic_checkpoints_and_baseline(self):
        with patch('matplotlib.figure.Figure.savefig') as save:
            fig = make_fixed_year_figure(self.fixed_year_payload())
            self.assertEqual(len(fig.axes), 6)
            np.testing.assert_array_equal(fig.axes[0].lines[-1].get_xdata(), [0, 5, 10])
            np.testing.assert_allclose(fig.axes[0].lines[-1].get_ydata(), [15., 21., 16.5])
            np.testing.assert_allclose(fig.axes[0].lines[-2].get_ydata(), [15., 15.])
            np.testing.assert_allclose(fig.axes[-1].lines[-1].get_ydata(), [13.5, 19.5, 15.])
            self.assertIn('in-sample', fig._suptitle.get_text())
            self.assertIn('2013', fig.axes[0].get_xlabel())
            fig.canvas.draw()
            save.assert_not_called()
            plt.close(fig)

    def test_fixed_year_plot_rejects_mixed_years_sampling_and_missing_baseline(self):
        for issue in ('year', 'deterministic', 'baseline', 'accounting', 'alignment', 'feasible'):
            payload = self.fixed_year_payload()
            row = payload['fixed_year_histories'][0][0]
            if issue == 'year':
                row['year'] = 2012
            elif issue == 'deterministic':
                row['deterministic'] = False
            elif issue == 'baseline':
                for history in payload['fixed_year_histories']:
                    history.pop(0)
            elif issue == 'accounting':
                row['total_reward'] += 1.
            elif issue == 'alignment':
                payload['fixed_year_histories'][1].pop()
            else:
                row['feasible'] = False
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                make_fixed_year_figure(payload)

    def test_fixed_year_plot_entry_point_does_not_require_validation_or_references(self):
        payload = self.fixed_year_payload()
        with patch('examples.plot_cascade.select_interactive_backend', return_value='Agg'), \
                patch('sys.argv', ['plot_cascade.py']), \
                patch('sys.stdin', io.StringIO(json.dumps(payload))), \
                patch('examples.plot_cascade.make_figure') as validation, \
                patch('examples.plot_cascade.make_validation_reward_figure') as reward, \
                patch('matplotlib.pyplot.show') as show:
            main()
            validation.assert_not_called()
            reward.assert_not_called()
            show.assert_called_once_with(block=True)
        plt.close('all')

    def validation_reward_payload(self):
        return dict(seeds=[1,2],validation_years=list(range(2020,2025)),
            histories=[[dict(episode=e,training_steps=steps,mean_reward=r,mean_TWh=100.)
                        for e,steps,r in zip((0,10,20),(0,3650,7300+i),rewards)]
                       for i,rewards in enumerate(((2.,6.,-2.),(4.,10.,-6.)))])

    def test_validation_reward_uses_recorded_returns_and_sample_sd(self):
        payload=self.validation_reward_payload()
        fig=make_validation_reward_figure(payload);ax=fig.axes[0]
        np.testing.assert_array_equal(ax.lines[-1].get_xdata(),[0,10,20])
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(),[3.,8.,-4.])
        for line,expected in zip(ax.lines[:2],([2.,6.,-2.],[4.,10.,-6.])):
            np.testing.assert_array_equal(line.get_ydata(),expected)
        vertices=ax.collections[0].get_paths()[0].vertices
        for x,mean,sd in zip((0,10,20),(3.,8.,-4.),np.sqrt(2.)*np.array([1.,2.,2.])):
            y=vertices[vertices[:,0]==x,1]
            self.assertAlmostEqual(y.min(),mean-sd)
            self.assertAlmostEqual(y.max(),mean+sd)
        self.assertIn('2020-2024',ax.get_title())
        self.assertIn('Undiscounted',ax.get_ylabel())
        self.assertIn('Seed +/-1 SD',ax.get_legend_handles_labels()[1])
        self.assertTrue(all(line.get_linestyle()!='--' for line in ax.lines))
        self.assertLess(ax.get_ylim()[0],-4.)
        plt.close(fig)

    def test_validation_reward_single_seed_and_invalid_inputs(self):
        payload=self.validation_reward_payload()
        single=dict(payload,seeds=[1],histories=payload['histories'][:1])
        fig=make_validation_reward_figure(single)
        self.assertEqual(len(fig.axes[0].collections),0)
        plt.close(fig)
        for issue in ('missing_reward','nonfinite','mismatched_checkpoints','empty','missing_seed'):
            bad=copy.deepcopy(payload)
            if issue=='missing_reward':del bad['histories'][0][0]['mean_reward']
            elif issue=='nonfinite':bad['histories'][0][0]['mean_reward']=float('nan')
            elif issue=='mismatched_checkpoints':bad['histories'][1][1]['episode']=11
            elif issue=='empty':bad['histories']=[[],[]]
            else:bad['seeds']=[1]
            with self.subTest(issue=issue),self.assertRaises(ValueError):
                make_validation_reward_figure(bad)

    def test_reward_labels_and_records_share_the_experiment_gamma(self):
        for gamma in (0., .5, 1.):
            payload = self.validation_reward_payload()
            payload['gamma'] = gamma
            for history in payload['histories']:
                for row in history:
                    row['gamma'] = gamma
            fig = make_validation_reward_figure(payload)
            self.assertIn('gamma={:g}'.format(gamma), fig.axes[0].get_ylabel())
            self.assertIn('Undiscounted' if gamma == 1. else 'Discounted', fig.axes[0].get_ylabel())
            plt.close(fig)
            payload['histories'][0][0]['gamma'] = .25
            with self.assertRaisesRegex(ValueError, 'same gamma'):
                make_validation_reward_figure(payload)

        payload = self.fixed_year_payload()
        payload['gamma'] = .5
        for history in payload['fixed_year_histories']:
            for row in history:
                row['gamma'] = .5
        fig = make_fixed_year_figure(payload)
        self.assertIn('TWh', fig.axes[0].get_ylabel())
        self.assertTrue(all('Discounted' in ax.get_ylabel() for ax in fig.axes[1:]))
        plt.close(fig)

        payload = self.reward_scale_payload()
        payload['gamma'] = .5
        for history in payload['episode_histories']:
            for row in history:
                row['gamma'] = .5
        figures = make_reward_scale_figures(payload)
        self.assertIn('gamma=0.5', figures[0].axes[0].get_ylabel())
        for fig in figures:
            plt.close(fig)

    def seasonal_payload(self):
        metadata=dict(seasonal_constraints=True,action_bins=37,head_model='storage_sensitivity',
                      head_data_status='assumed_sensitivity_only',terminal_storage_mode='initial',
                      feasible=True,violation_count=0)
        return dict(seeds=[1],validation_years=[2020],test_years=[1998],test_labels=['wet'],
            histories=[[dict(episode=0,mean_TWh=100.,years=[dict(year=2020,energy_TWh=100.,**metadata)])]],
            references=[dict(year=y,energy_TWh=90.,reference_type='seasonal_target_tracking',**metadata)
                        for y in (2020,1998)],
            tests=[dict(year=1998,typical_label='wet',seed=1,energy_TWh=95.,**metadata)],
            reference_type='seasonal_target_tracking',reference_label='Seasonal rule reference',**metadata)

    def test_custom_two_year_validation_and_single_test_plot(self):
        payload = self.seasonal_payload()
        metadata = dict(payload['references'][0], gamma=.99)
        payload.update(seeds=[1, 2, 3], gamma=.99, validation_years=[2012, 2013],
                       test_years=[2014], test_labels=['test_2014'])
        payload['references'] = [dict(metadata, year=y) for y in (2012, 2013, 2014)]
        payload['histories'] = [[dict(episode=0, mean_TWh=100.+seed,
            mean_reward=10.+seed, gamma=.99,
            years=[dict(metadata, year=y) for y in (2012, 2013)])] for seed in payload['seeds']]
        payload['tests'] = [dict(metadata, year=2014, typical_label='test_2014',
                                 seed=seed) for seed in payload['seeds']]
        with patch('matplotlib.figure.Figure.savefig') as save:
            fig = make_figure(payload)
            reward = make_validation_reward_figure(payload)
            fig.canvas.draw()
            reward.canvas.draw()
            self.assertIn('2012-2013', fig.axes[0].get_title())
            self.assertIn('2014', fig.axes[1].get_xticklabels()[0].get_text())
            self.assertEqual(len(fig.axes[1].patches), 2)
            self.assertIn('2012-2013', reward.axes[0].get_title())
            save.assert_not_called()
            plt.close(fig)
            plt.close(reward)

    def test_seasonal_reference_labels_and_shared_comparison_mode(self):
        payload=self.seasonal_payload()
        fig=make_figure(payload)
        self.assertNotIn('Seasonal rule reference',fig.axes[0].get_legend_handles_labels()[1])
        self.assertIn('Seasonal rule reference',fig.axes[1].get_legend_handles_labels()[1])
        self.assertIn('seasonal constraints',fig._suptitle.get_text())
        self.assertEqual(list(fig.axes[0].lines[-1].get_ydata()),[100.])
        self.assertEqual(fig.axes[1].patches[-1].get_height(),90.)
        plt.close(fig)
        for change in ('seasonal_constraints','action_bins','head_model','head_data_status',
                       'terminal_storage_mode','feasible','violation_count'):
            bad=copy.deepcopy(payload)
            bad['references'][0][change]={'seasonal_constraints':False,'action_bins':200,
                'head_model':'fixed','head_data_status':'other','terminal_storage_mode':'free',
                'feasible':False,'violation_count':1}[change]
            with self.subTest(field=change),self.assertRaises(ValueError):make_figure(bad)
        for bad in (dict(payload,references=payload['references'][:1]),
                    dict(payload,references=payload['references']*2),
                    dict(payload,reference_type='constant_storage')):
            with self.assertRaises(ValueError):make_figure(bad)

    def test_detail_windows_reward_units_smoothing_and_single_show(self):
        histories=[]
        for seed in (1,2):
            rows=[]
            for episode,energy in enumerate([1.,2.,4.],1):
                row=dict(iteration=episode,episode_energy_TWh=energy*seed,
                         critic_value_loss=episode*.1)
                for i in range(5):
                    row.update({'actor{}_policy_loss'.format(i):-episode*.01,
                                'actor{}_dist_entropy'.format(i):5.-episode*.1,
                                'actor{}_actor_grad_norm'.format(i):episode*.2})
                rows.append(row)
            histories.append(rows)
        payload=dict(seeds=[1,2],training_histories=histories,reference_energy_MWh=2e6,
                     histories=[[dict(episode=0,mean_TWh=1.),dict(episode=3,mean_TWh=2.)]]*2,
                     references=[dict(year=2020,energy_TWh=3.)],validation_years=[2020],
                     tests=[],test_years=[],test_labels=[])
        with patch('matplotlib.figure.Figure.savefig') as save:
            figures=make_training_figures(payload,window=2)
            self.assertEqual(len(figures),3)
            self.assertEqual([len(fig.axes) for fig in figures],[1,1,4])
            self.assertEqual(list(figures[0].axes[0].lines[-1].get_ydata()),[.75,1.125,2.25])
            self.assertEqual(list(figures[1].axes[0].lines[-1].get_ydata()),[1.5,2.25,4.5])
            for fig in figures:plt.close(fig)
            with patch('examples.plot_cascade.select_interactive_backend',return_value='Agg'), \
                 patch('sys.argv',['plot_cascade.py','--plot-details']), \
                 patch('sys.stdin',io.StringIO(json.dumps(payload))),patch('matplotlib.pyplot.show') as show:
                main()
                self.assertEqual(len(plt.get_fignums()),4)
                show.assert_called_once_with(block=True)
            save.assert_not_called()
            plt.close('all')
        payload['training_histories'][1]=histories[1][:-1]
        with self.assertRaisesRegex(ValueError,'matching training episodes'):
            make_training_figures(payload)

    def test_penalized_comparison_still_plots_generation_and_rejects_mixed_costs(self):
        payload=self.seasonal_payload()
        penalties=dict(enabled=True,target_band_weight=.1,daily_rise_weight=.1,daily_fall_weight=.1)
        payload['reward_penalties']=penalties
        rows=payload['references']+payload['tests']+payload['histories'][0][0]['years']
        for row in rows:row.update(reward_penalties=dict(penalties),episode_reward=1.)
        fig=make_figure(payload)
        self.assertIn('selected by penalized return',fig._suptitle.get_text())
        self.assertEqual(list(fig.axes[0].lines[-1].get_ydata()),[100.])
        self.assertTrue(all(line.get_linestyle()!='--' for line in fig.axes[0].lines))
        self.assertEqual([bar.get_height() for bar in fig.axes[1].patches],[95.,90.])
        plt.close(fig)
        for change in ({'enabled':False},dict(penalties,target_band_weight=.2)):
            bad=copy.deepcopy(payload);bad['references'][0]['reward_penalties']=change
            with self.assertRaisesRegex(ValueError,'same reward penalties'):make_figure(bad)
        payload['reward_penalties']={'enabled':False}
        with self.assertRaisesRegex(ValueError,'same reward penalties'):make_figure(payload)

    def test_validation_reward_window_is_opened_with_and_without_plot_details(self):
        payload=self.seasonal_payload()
        penalties=dict(enabled=True,target_band_weight=.1,daily_rise_weight=.1,daily_fall_weight=.1)
        payload['reward_penalties']=penalties
        payload['histories'][0][0]['mean_reward']=7.
        for row in payload['references']+payload['tests']+payload['histories'][0][0]['years']:
            row.update(reward_penalties=dict(penalties),episode_reward=7.)
        metrics=dict(iteration=1,episode_energy_TWh=10.,episode_reward=3.,critic_value_loss=.1)
        for i in range(5):
            metrics.update({'actor{}_policy_loss'.format(i):-.01,
                            'actor{}_dist_entropy'.format(i):1.,
                            'actor{}_actor_grad_norm'.format(i):.2})
        payload.update(training_histories=[[metrics]],reference_energy_MWh=1624800.)
        for details in (False,True):
            plt.close('all')
            argv=['plot_cascade.py']+(['--plot-details'] if details else [])
            with self.subTest(details=details),patch('matplotlib.figure.Figure.savefig') as save, \
                 patch('examples.plot_cascade.select_interactive_backend',return_value='Agg'), \
                 patch('sys.argv',argv),patch('sys.stdin',io.StringIO(json.dumps(payload))), \
                 patch('matplotlib.pyplot.show') as show:
                main()
                self.assertEqual(len(plt.get_fignums()),5 if details else 2)
                titles=[plt.figure(n).axes[0].get_title() for n in plt.get_fignums()]
                self.assertIn('2020-2020 mean annual validation net reward',titles)
                show.assert_called_once_with(block=True)
                save.assert_not_called()
            plt.close('all')

    def test_reward_plot_uses_actual_penalized_return_and_requires_it(self):
        rows=[]
        for episode,energy,reward in ((1,10.,3.),(2,20.,-1.)):
            row=dict(iteration=episode,episode_energy_TWh=energy,episode_reward=reward,critic_value_loss=.1)
            for i in range(5):
                row.update({'actor{}_policy_loss'.format(i):-.01,
                            'actor{}_dist_entropy'.format(i):1.,
                            'actor{}_actor_grad_norm'.format(i):.2})
            rows.append(row)
        payload=dict(seeds=[1],training_histories=[rows],reference_energy_MWh=1e6,
                     reward_penalties={'enabled':True})
        figures=make_training_figures(payload,window=1)
        self.assertEqual(list(figures[0].axes[0].lines[-1].get_ydata()),[3.,-1.])
        self.assertEqual(list(figures[1].axes[0].lines[-1].get_ydata()),[10.,20.])
        self.assertIn('penalized return',figures[0].axes[0].get_title())
        for fig in figures:plt.close(fig)
        del rows[0]['episode_reward']
        with self.assertRaisesRegex(ValueError,'actual episode_reward'):make_training_figures(payload)

    def test_two_panels_and_no_save_even_when_day_counts_differ(self):
        years = [1998, 1999, 1970, 1997, 2006]
        labels = ["extremely_wet", "wet", "normal", "dry", "extremely_dry"]
        payload = dict(seeds=[1,2], validation_years=list(range(2020,2025)),
            test_years=years, test_labels=labels,
            histories=[[dict(episode=0,mean_TWh=100.,training_steps=0),
                        dict(episode=2,mean_TWh=101.,training_steps=steps)] for steps in [730,731]],
            references=[dict(year=y,energy_TWh=99.) for y in years+list(range(2020,2025))],
            tests=[dict(year=y,typical_label=label,seed=s,energy_TWh=102.)
                   for y,label in zip(years,labels) for s in [1,2]])
        with patch("matplotlib.figure.Figure.savefig") as save:
            fig = make_figure(payload)
            self.assertEqual(len(fig.axes), 2)
            self.assertIn("episodes", fig.axes[0].get_xlabel())
            self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [0,2])
            self.assertEqual(len(fig.axes[1].patches), 10)
            save.assert_not_called()
            plt.close(fig)
            payload['tests']=[]
            fig=make_figure(payload)
            self.assertFalse(fig.axes[1].axison)
            self.assertIn('skipped',fig.axes[1].texts[0].get_text())
            save.assert_not_called()
            plt.close(fig)

    def reward_scale_payload(self):
        histories=[]
        for seed in (1,2):
            rows=[]
            for episode,power,band,rise,fall in [(1,10.,1.,.5,0.),(2,0.,1.,0.,0.),(3,30.,3.,0.,1.5)]:
                rows.append(dict(seed=seed,episode=episode,power_reward=power*seed,
                    band_penalty=band*seed,rise_penalty=rise*seed,fall_penalty=fall*seed,
                    total_reward=(power-band-rise-fall)*seed))
            histories.append(rows)
        return dict(seeds=[1,2],episode_histories=histories)

    def test_reward_components_and_energy_weighted_percentage_windows(self):
        with patch('matplotlib.figure.Figure.savefig') as save:
            figures=make_reward_scale_figures(self.reward_scale_payload(),window=2)
            components,ratios=figures
            self.assertEqual([ax.get_title() for ax in components.axes[:5]],
                             ['Power Reward','Band penalty','Rise penalty','Fall penalty','Total Reward'])
            np.testing.assert_allclose(components.axes[0].lines[-1].get_ydata(),[15.,7.5,22.5])
            np.testing.assert_allclose(components.axes[1].lines[-1].get_ydata(),[1.5,1.5,3.])
            # Episode 2 has no generation but still incurs a cost. Its raw
            # percentage is undefined; the 2-episode ratio of sums is 20%.
            self.assertTrue(np.isnan(ratios.axes[0].lines[0].get_ydata()[1]))
            np.testing.assert_allclose(ratios.axes[0].lines[-1].get_ydata(),[10.,20.,100.*4/30])
            np.testing.assert_allclose(ratios.axes[3].lines[-1].get_ydata(),[15.,25.,100.*5.5/30])
            for fig in figures:fig.canvas.draw();plt.close(fig)
            save.assert_not_called()

    def test_zero_power_ratios_are_gaps_and_invalid_accounting_is_rejected(self):
        payload=self.reward_scale_payload()
        for rows in payload['episode_histories']:
            for row in rows:
                row['power_reward']=0.
                row['total_reward']=-sum(row[k+'_penalty'] for k in ('band','rise','fall'))
        figures=make_reward_scale_figures(payload)
        self.assertTrue(np.isnan(figures[1].axes[0].lines[-1].get_ydata()).all())
        for fig in figures:fig.canvas.draw();plt.close(fig)
        for issue in ('nonfinite','negative','accounting','seed','alignment'):
            bad=self.reward_scale_payload();row=bad['episode_histories'][0][0]
            if issue=='nonfinite':row['power_reward']=np.inf
            elif issue=='negative':row['band_penalty']=-1.
            elif issue=='accounting':row['total_reward']+=1.
            elif issue=='seed':row['seed']=7
            else:bad['episode_histories'][1].pop()
            with self.subTest(issue=issue),self.assertRaises(ValueError):make_reward_scale_figures(bad)

    def test_reward_scaling_windows_open_without_plot_details(self):
        payload=self.seasonal_payload()
        payload['episode_histories']=self.reward_scale_payload()['episode_histories'][:1]
        with patch('examples.plot_cascade.select_interactive_backend',return_value='Agg'), \
             patch('sys.argv',['plot_cascade.py']),patch('sys.stdin',io.StringIO(json.dumps(payload))), \
             patch('matplotlib.pyplot.show') as show,patch('matplotlib.figure.Figure.savefig') as save:
            main()
            self.assertEqual(len(plt.get_fignums()),3)
            show.assert_called_once_with(block=True);save.assert_not_called()
        plt.close('all')


if __name__ == "__main__":
    unittest.main()
