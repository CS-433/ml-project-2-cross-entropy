import abc

class BaseMethod:
    def __init__(self, dataloader):
        self.pre_process(dataloader)
        self.dataloader = dataloader

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    def pre_process(self, dataloader):
        pass