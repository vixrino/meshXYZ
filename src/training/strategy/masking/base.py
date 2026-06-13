from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jaxtyping import Bool
from torch import Tensor

if TYPE_CHECKING:
    from ....dataset.types import Batch


class BaseMasking(ABC):
    @abstractmethod
    def select(self, batch: "Batch") -> Bool[Tensor, "batch query_faces key_faces"]:
        """Per-query attention mask: True means key face is hidden for that query."""
