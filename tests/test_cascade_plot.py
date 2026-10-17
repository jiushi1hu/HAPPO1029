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
from examples.plot_cascade import make_figure, make_training_figures, main


class PlotTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
