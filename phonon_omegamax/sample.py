from dataclasses import dataclass

from pymatgen.core import Structure


@dataclass(frozen=True)
class Sample:
    mp_id: str
    structure: Structure
    target: float

    def __post_init__(self):
        if not (self.target > 0 and self.target == self.target):  # finite + > 0
            raise ValueError(f"target must be positive finite, got {self.target}")
