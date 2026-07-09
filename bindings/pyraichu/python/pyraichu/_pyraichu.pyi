"""Type stubs for the Rust extension module ``pyraichu._pyraichu``."""

__version__: str

class ModelError(Exception): ...
class SimulationError(Exception): ...

def validate_model(model_json: str) -> None: ...
def simulate_json(
    model_json: str,
    t_max: float,
    journal: bool = False,
    confluence_check: bool = False,
    samples: list[float] | None = None,
    seed: int = 0,
    rng_stream: int = 0,
) -> str: ...
def monte_carlo_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    samples: list[float],
    seed: int = 0,
    threads: int | None = None,
    quantiles: list[float] | None = None,
) -> str: ...
