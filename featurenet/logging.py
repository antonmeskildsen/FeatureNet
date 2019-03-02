import json
from typing import List, Dict

import pandas as pd
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import accuracy


class ConfusionMatrix(accuracy._BaseClassification):

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if self._type == "binary":
            y_pred = torch.round(y_pred).view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            raise ValueError('Multiclass not yet supported')

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        tp = (y * y_pred).sum()
        fp = y_pred.sum() - tp

        y_pred_neg = y_pred.neg() + 1
        y_neg = y.neg() + 1

        tn = (y_neg * y_pred_neg).sum()
        fn = y_pred_neg.sum() - tn

        self.confusion_matrix += torch.tensor([[tp, fp], [fn, tn]])

    def compute(self):
        if not isinstance(self.confusion_matrix, torch.Tensor):
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed".format(self.__class__.__name__))

        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = torch.zeros(2, 2)


class Logger:

    def __init__(self, metrics: List[str], name: str):
        self.data = {m: [] for m in metrics}
        self.name = name

    def __getitem__(self, item):
        return self.data[item]

    def log_step(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.data[k].append(v)

    def save(self, path):
        j = json.dumps(self.data)
        with open(path, 'w') as file:
            file.write(j)

    def to_html(self):
        frame = pd.DataFrame.from_dict(self.data)
        td = {
            'selector': 'td',
            'props': [
                ('border-style', 'solid'),
                ('border-width', '1px'),
                ('padding', '5px')
            ]
        }
        styles = [td]
        st = frame.style
        st.set_table_styles(styles)
        return st.render()