"""Config loading tests."""

import pytest
import yaml

from eonet_cascades.config import DataConfig, load_data_config


def test_data_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EONET_DATA_ROOT", str(tmp_path))
    cfg = DataConfig()
    assert cfg.data_root == tmp_path
    assert cfg.duckdb_path == tmp_path / "events.duckdb"
    assert cfg.raw_dir == tmp_path / "raw"
    assert cfg.manifests_dir == tmp_path / "manifests"
    assert "eonet" in cfg.catalogs
    assert cfg.bbox == (-130.0, 14.0, -65.0, 50.0)  # CONUS + Mexico


def test_load_data_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "conus.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "alt"),
                "catalogs": ["eonet", "usgs"],
                "bbox": [-130.0, 14.0, -65.0, 50.0],
                "start_date": "2000-01-01",
            }
        )
    )
    cfg = load_data_config(yaml_path)
    assert cfg.data_root == tmp_path / "alt"
    assert cfg.catalogs == ["eonet", "usgs"]


def test_data_config_missing_drive_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("EONET_DATA_ROOT", str(tmp_path / "does-not-exist"))
    cfg = DataConfig()
    with pytest.raises(FileNotFoundError, match="data_root does not exist"):
        cfg.ensure_exists()
