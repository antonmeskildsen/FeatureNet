from abc import ABC, abstractmethod

class MetricFunc(ABC):

    @abstractmethod
    def __call__(self, input, output):
        ...

class AccuracyFunc(MetricFunc):

    def __call__(self, input, output):


class Logger:

    def __init__(self, ):

    def log_step(self, input, output):
        ...