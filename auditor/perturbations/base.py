from abc import ABC, abstractmethod, abstractproperty
from typing import List


class AbstractPerturbation(ABC):
    """Abstract class to aid in creation of perturbation classes
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def perturb(self) -> List[str]:
        raise NotImplementedError(
            'Derived class must override the perturb method.'
        )

    @abstractproperty
    def description(self):
        pass
