from typing import TypedDict

from jaxtyping import Bool, Float, Int
from torch import Tensor


class Batch(TypedDict):
    pc:             Float[Tensor, "batch points 3"]
    faces:          Int[Tensor, "batch faces 9"]
    lengths:        Int[Tensor, "batch"]
    face_neighbors: Int[Tensor, "batch faces 3"]
