"""CLI smoke tests."""

from typer.testing import CliRunner

from eonet_cascades.cli import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eonet" in result.stdout.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.stdout
