from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class DatasetLoader(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_metadata(self, index):
        return None  # Optional override
