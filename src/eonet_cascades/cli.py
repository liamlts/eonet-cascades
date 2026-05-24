"""Top-level Typer CLI for the eonet-cascades project."""

from __future__ import annotations

import typer
from rich.console import Console

from eonet_cascades import __version__

app = typer.Typer(
    name="eonet",
    help="Spatio-temporal point process benchmark for natural-hazard event cascades.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """EONET Cascades CLI."""
    if ctx.invoked_subcommand is None:
        console.print(__version__)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
