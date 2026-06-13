from typing import TypedDict

from jaxtyping import Bool, Float, Int
from torch import Tensor


class Batch(TypedDict):
    pc:             Float[Tensor, "batch points 3"]
    faces:          Int[Tensor, "batch faces coords"]    # coords=9 (tri) or 12 (quad/mixed)
    lengths:        Int[Tensor, "batch"]
    face_neighbors: Int[Tensor, "batch faces slots"]     # slots=3 (tri) or 4 (quad/mixed)
