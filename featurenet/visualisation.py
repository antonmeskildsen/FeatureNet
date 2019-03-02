import time
import visdom

import numpy as np

from abc import ABC
from typing import List

from featurenet.logging import Logger
from featurenet.helpers import time_code_id


class Window(ABC):
    def __init__(self, env_name):
        if env_name is None:
            env_name = time_code_id()
        self.env_name = env_name
        self.win = None
        self.vis = visdom.Visdom(env=self.env_name)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        self.vis.close(self.win)


class MetricPlot(Window):
    def __init__(self, loggers: List[Logger], metric: str, units='Epoch', line_colors=None, env_name=None):
        super().__init__(env_name)
        if line_colors is None:
            line_colors = np.array([255, 0, 255])
        self.line_colors = line_colors
        self.loggers = loggers
        self.metric = metric
        self.units = units
        self.legend = [m.name for m in self.loggers]

    def update(self):
        data = np.vstack([m[self.metric] for m in self.loggers]).T
        x = np.linspace(1, data.shape[0], data.shape[0])
        self.win = self.vis.line(
            data,
            x,
            win=self.win,
            update='replace' if self.win else None,
            opts=dict(
                legend=self.legend,
                markers=True,
                xlabel=self.units,
                ylabel=self.metric,
                title=self.metric,
            )
        )
        time.sleep(0.01)


class Image(Window):

    def __init__(self, env_name=None):
        super().__init__(env_name)

    def update(self, img):
        self.win = self.vis.image(
            img,
            win=self.win,
        )


class Images(Window):

    def __init__(self, num_cols, padding=5, env_name=None):
        super().__init__(env_name)
        self.num_cols = num_cols
        self.padding = padding

    def update(self, img_list):
        self.win = self.vis.images(
            img_list,
            self.num_cols,
            self.padding,
            win=self.win
        )


class Html(Window):

    def update(self, html):
        self.win = self.vis.text(
            html,
            win=self.win
        )


def properties(env_name, props):
    def decorator(callback):
        vis = visdom.Visdom(env=env_name)
        win = vis.properties(props)
        vis.register_event_handler(callback, win)
    return decorator
