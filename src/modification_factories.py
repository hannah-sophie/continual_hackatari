import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class ModificationFactory(ABC):

    def __init__(self, num_total_steps):
        self.num_total_steps = num_total_steps
    
    @abstractmethod
    def get_modification(self, step):
        """
        Returns a modification string for the given step.
        :param step: The current step in the game.
        :return: A string representing the modification to apply.
        """
        pass

    def get_total_timesteps(self):
        return self.num_total_steps

class NoModificationFactory(ModificationFactory):
    """
    A modification factory that does not apply any modifications.
    """

    def get_modification(self, step):
        return ""


class SequentialModificationFactory(ModificationFactory):
    """
    A modification factory that applies modifications sequentially.
    """

    def __init__(self, num_total_steps, modifications: List[str], switching_thresholds: List[int]):
        super().__init__(num_total_steps)
        assert len(modifications)-1 == len(switching_thresholds), "Number of modifications must match number of switching thresholds minus one."
        assert switching_thresholds == sorted(switching_thresholds), "Switching thresholds must be sorted in ascending order."
        self.modifications = modifications
        switching_thresholds = np.append(np.array(switching_thresholds), num_total_steps)
        self.switching_thresholds = switching_thresholds

    def get_modification(self, step):
        assert step < self.num_total_steps, "Step must be less than the total number of steps."
        idx = np.where(step < self.switching_thresholds)[0][0]
        return self.modifications[idx]


class RandomModificationFactory(ModificationFactory):
    """
    A modification factory that applies modifications randomly.
    """

    def __init__(self, num_total_steps, modifications: List[str], num_repetitions: int = 1, seed=None):
        super().__init__(num_total_steps)
        self.modifications = modifications
        self.rng = np.random.default_rng(seed)
        self.num_repetitions = num_repetitions
        self.current_repetitions = 0
        self.current_modification = self.rng.choice(self.modifications)

    def get_modification(self, step):
        assert step < self.num_total_steps, "Step must be less than the total number of steps."
        if self.current_repetitions >= self.num_repetitions:
            self.current_modification = self.rng.choice(self.modifications)
            self.current_repetitions = 0
        else:
            self.current_repetitions += 1
        return self.current_modification


def get_modification_factory(
    modification_factory_name: str, modification_factory_kwargs: Dict[str, Any]
) -> ModificationFactory:
    modification_factory = modification_factory_mapping.get(modification_factory_name)
    return modification_factory(**modification_factory_kwargs)


modification_factory_mapping = {"NoModificationFactory": NoModificationFactory,
                                "SequentialModificationFactory": SequentialModificationFactory,
                                "RandomModificationFactory": RandomModificationFactory}
