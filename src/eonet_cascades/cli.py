"""Top-level Typer CLI for the eonet-cascades project."""

from __future__ import annotations

import typer
from rich.console import Console

from eonet_cascades import __version__

app = typer.Typer(
    name="eonet",
    help="Spatio-temporal point process benchmark for natural-hazard event cascades.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Root callback — disables Typer's single-command flattening so
    `eonet --help` shows the app help (not the only subcommand's help) until
    Task 16 adds a second command."""


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)
