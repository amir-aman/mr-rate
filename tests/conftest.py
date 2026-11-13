from pathlib import Path

import pytest

from mrrate_vlm.config import Config
from mrrate_vlm.synthetic import create_fixture


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    create_fixture(tmp_path, 6)
    return tmp_path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        data_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        bootstrap_samples=100,
        max_studies=6,
        source_batches=(),
    )


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
