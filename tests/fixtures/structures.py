"""Small deterministic crystal-structure fixtures for tests.

These structures are real but tiny: NaCl rocksalt, MgO rocksalt, GaAs zincblende.
Enough variety to exercise graph construction and Magpie featurization without
requiring matbench data downloads.
"""
from __future__ import annotations

from pymatgen.core import Lattice, Structure

from phonon_omegamax.sample import Sample


def nacl() -> Structure:
    lat = Lattice.cubic(5.64)
    return Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def mgo() -> Structure:
    lat = Lattice.cubic(4.21)
    return Structure(lat, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def gaas() -> Structure:
    lat = Lattice.cubic(5.65)
    return Structure(
        lat, ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )


def fake_dataset(n: int = 10) -> list[Sample]:
    """Repeat the three fixture structures with varied targets."""
    structures = [nacl(), mgo(), gaas()]
    samples: list[Sample] = []
    for i in range(n):
        s = structures[i % len(structures)]
        # Spread targets across a realistic ω_max range (50–1500 cm⁻¹).
        target = 100.0 + (i * 137.0) % 1400
        samples.append(Sample(mp_id=f"mp-fake-{i}", structure=s, target=target))
    return samples
