from abc import ABC, abstractmethod, abstractproperty
from typing import List


class TransformBase(ABC):
    """Base class to aid in creation of transformations
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(
        self,
        prompt: str,
        *args,
        **kwargs,
    ) -> List[str]:
        """Method to generate transformations. The method must except an
        argument 'prompt' of string type.

        Raises:
            NotImplementedError: Riased when derived class must implement
                this method.

        Returns:
            List[str]: Must return a list of strings.
        """
        raise NotImplementedError(
            'Derived class must override the tranform method.'
        )

    @abstractproperty
    def description(self):
        pass
