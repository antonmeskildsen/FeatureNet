import pytest

from featurenet.visualisation import MetricPlot
from featurenet.logging import Logger


class TestMetricPlots:

    a = [2.3, 5.0, 0.5, 0.7]
    b = [1.5, 1.6, 2.1, 2.3]

    def test_simple_valid_plot(self):
        la = Logger(['m'], 'train')
        lb = Logger(['m'], 'val')

        for e in self.a:
            la.log_step({'m': e})

        for e in self.b:
            lb.log_step({'m': e})

        plot = MetricPlot([la, lb], 'm', env_name='test')
        plot.update()

        plot.vis.delete_env(plot.env_name)