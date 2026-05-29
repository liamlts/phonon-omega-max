import pytest

from tests.fixtures.structures import fake_dataset, nacl, mgo, gaas


@pytest.fixture
def nacl_structure():
    return nacl()


@pytest.fixture
def mgo_structure():
    return mgo()


@pytest.fixture
def gaas_structure():
    return gaas()


@pytest.fixture
def fake_samples():
    return fake_dataset(n=12)
