import datetime
import time
import visdom

from abc import ABC


class Window(ABC):
    def __init__(self, env_name):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.win = None
        self.vis = visdom.Visdom(env=self.env_name)

    def update(self, *args, **kwargs):
        pass

    def close(self):
        self.vis.close(self.win)


class Plot(Window):
    def __init__(self, title, y_label, x_label, env_name=None):
        super().__init__(env_name)
        self.title = title
        self.y_label = y_label
        self.x_label = x_label

    def update(self, x, y):
        self.win = self.vis.line(
            [x],
            [y],
            win=self.win,
            update='append' if self.win else None,
            opts=dict(
                xlabel=self.x_label,
                ylabel=self.y_label,
                title=self.title,
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
