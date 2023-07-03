from abc import abstractmethod
from typing_extensions import override


class Feature:
    @abstractmethod
    def distance_to(self, another_feature) -> float:
        return None


class PointFeature(Feature):
    def __init__(self, keypoint, descriptor: None) -> None:
        super().__init__()
        self.keypoint = keypoint
        self.desc = descriptor
        self.img = None
