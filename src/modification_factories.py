import itertools
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


class EpsSequentialModificationFactory(ModificationFactory):
    """
    A modification factory that applies modifications sequentially but selects a random previously seen modification with probability epsilon.
    """

    def __init__(
        self,
        num_total_steps,
        modifications: List[str],
        switching_thresholds: List[int],
        epsilon: float = 0.05,
    ):
        super().__init__(num_total_steps)
        assert len(modifications) - 1 == len(
            switching_thresholds
        ), "Number of modifications must match number of switching thresholds minus one."
        assert switching_thresholds == sorted(
            switching_thresholds
        ), "Switching thresholds must be sorted in ascending order."
        self.modifications = modifications
        switching_thresholds = np.append(
            np.array(switching_thresholds), num_total_steps
        )
        self.switching_thresholds = switching_thresholds
        self.epsilon = epsilon

    def get_modification(self, step):
        assert (
            step < self.num_total_steps
        ), "Step must be less than the total number of steps."
        idx = np.where(step < self.switching_thresholds)[0][0]
        if idx > 0 and np.random.rand() < self.epsilon:
            # Select a random previously seen modification
            idx = np.random.randint(0, idx)
        return self.modifications[idx]


class SequentialModificationFactory(EpsSequentialModificationFactory):
    """
    A modification factory that applies modifications sequentially.
    """

    def __init__(
        self, num_total_steps, modifications: List[str], switching_thresholds: List[int]
    ):
        super().__init__(
            num_total_steps, modifications, switching_thresholds, epsilon=0
        )


class RandomModificationFactory(ModificationFactory):
    """
    A modification factory that applies modifications randomly.
    """

    def __init__(
        self,
        num_total_steps,
        modifications: List[str],
        probabilities: List[float] = None,
        num_repetitions: int = 1,
    ):
        super().__init__(num_total_steps)
        if probabilities is None:
            probabilities = [1.0 / len(modifications)] * len(modifications)

        assert len(modifications) == len(
            probabilities
        ), "Number of modifications must match number of probabilities."
        assert np.isclose(sum(probabilities), 1.0), "Probabilities must sum to 1."
        self.modifications = modifications
        self.num_repetitions = num_repetitions
        self.current_repetitions = 0
        self.current_modification = np.random.choice(self.modifications)
        self.probabilities = probabilities

    def get_modification(self, step):
        assert (
            step < self.num_total_steps
        ), "Step must be less than the total number of steps."
        if self.current_repetitions >= self.num_repetitions:
            self.current_modification = np.random.choice(
                self.modifications, p=self.probabilities
            )
            self.current_repetitions = 0
        else:
            self.current_repetitions += 1
        return self.current_modification


def get_modification_factory(
    modification_factory_name: str, modification_factory_kwargs: Dict[str, Any]
) -> ModificationFactory:
    modification_factory = modification_factory_mapping.get(modification_factory_name)
    return modification_factory(**modification_factory_kwargs)


class AllCombinationsRandomModificationFactory(RandomModificationFactory):
    "ModificationFactory selecting for multiple sets of random modifications all combinations of these sets randomly"

    def __init__(
        self,
        num_total_steps,
        modifications: List[List[str]],
        probabilities: List[float] = None,
        num_repetitions: int = 1,
    ):
        super().__init__(
            num_total_steps,
            [" ".join(list(m)) for m in itertools.product(*modifications)],
            probabilities,
            num_repetitions,
        )


class EpsCombinedModificationFactory(ModificationFactory):
    """
    A modification factory where one can combine several base modification factories. Previously completed modifs are selected randomly.
    """

    def __init__(
        self,
        modification_factory_kwargs: Dict[str, dict],
        epsilon: float = 0.05,
    ):
        super().__init__(
            sum(
                [
                    modification_factory_kwargs[key]["num_total_steps"]
                    for key in modification_factory_kwargs
                ]
            )
        )
        self.all_modification_kwargs = modification_factory_kwargs
        self.all_modification_factories = list(modification_factory_kwargs.keys())
        self.completed_steps, self.completed_modifs = 0, []
        self.current_modification, self.current_modification_factory = None, None

        self.epsilon = epsilon

        self.set_current_modification_factory()

    def set_current_modification_factory(self):
        if self.current_modification is not None:
            self.completed_steps += self.all_modification_kwargs[
                self.current_modification
            ]["num_total_steps"]
            self.completed_modifs += self.all_modification_kwargs[
                self.current_modification
            ]["modifications"]

        self.current_modification = self.all_modification_factories.pop(0)
        assert (
            "CombinedModificationFactory" not in self.current_modification
        ), "No deep nesting of ModificationFactories allowed"
        self.current_modification_factory = get_modification_factory(
            self.current_modification,
            self.all_modification_kwargs[self.current_modification],
        )

    def get_modification(self, step):
        assert (
            step < self.num_total_steps
        ), "Step must be less than the total number of steps."
        step = step - self.completed_steps
        if step > self.current_modification_factory.num_total_steps:
            self.set_current_modification_factory()
            step = step - self.completed_steps

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.completed_modifs)
        return self.current_modification_factory.get_modification(step)


class CombinedModificationFactory(EpsCombinedModificationFactory):
    """
    A modification factory where one can combine several base modification factories.
    """

    def __init__(self, modification_factory_kwargs: Dict[str, dict]):
        super().__init__(modification_factory_kwargs, epsilon=0)


modification_factory_mapping = {
    "NoModificationFactory": NoModificationFactory,
    "EpsSequentialModificationFactory": EpsSequentialModificationFactory,
    "SequentialModificationFactory": SequentialModificationFactory,
    "RandomModificationFactory": RandomModificationFactory,
    "AllCombinationsRandomModificationFactory": AllCombinationsRandomModificationFactory,
    "EpsCombinedModificationFactory": EpsCombinedModificationFactory,
    "CombinedModificationFactory": CombinedModificationFactory,
}
