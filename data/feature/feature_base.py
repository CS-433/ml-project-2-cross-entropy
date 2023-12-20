import abc
from rascaline import AtomicComposition

class FeatureBase:
    @abc.abstractmethod
    def featurize(self, raw_data: AtomicComposition, file_name: str = None):
        pass