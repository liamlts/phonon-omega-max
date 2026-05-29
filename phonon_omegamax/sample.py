import math
from dataclasses import dataclass

from pymatgen.core import Structure


@dataclass(frozen=True)
class Sample:
    mp_id: str
    structure: Structure
    target: float  # ω_max in cm⁻¹

    def __post_init__(self):
        if not (math.isfinite(self.target) and self.target > 0):
            raise ValueError(
                f"target must be positive finite, got {self.target}"
            )
