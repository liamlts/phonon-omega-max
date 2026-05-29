import pytest
from pymatgen.core import Lattice, Structure

from phonon_omegamax.sample import Sample


def _rocksalt():
    lat = Lattice.cubic(5.64)
    return Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_sample_holds_required_fields():
    s = Sample(mp_id="mp-22862", structure=_rocksalt(), target=412.0)
    assert s.mp_id == "mp-22862"
    assert s.target == pytest.approx(412.0)
    assert len(s.structure) == 2


def test_sample_rejects_non_positive_target():
    with pytest.raises(ValueError):
        Sample(mp_id="mp-0", structure=_rocksalt(), target=0.0)
    with pytest.raises(ValueError):
        Sample(mp_id="mp-0", structure=_rocksalt(), target=-1.0)


def test_sample_is_frozen():
    s = Sample(mp_id="mp-22862", structure=_rocksalt(), target=412.0)
    with pytest.raises((AttributeError, TypeError)):
        s.target = 999.0
