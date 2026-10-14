"""In-memory plotting: episode alignment and no exported figure files."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples.plot_cascade import make_figure


class PlotTests(unittest.TestCase):
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
