"""Type stubs for the Rust extension module ``pyraichu._pyraichu``."""

__version__: str

class ModelError(Exception): ...
class SimulationError(Exception): ...

class FlowConfig:
    """Convergence policy of the continuous flow resolution, passed to
    every entry point under a single ``flow=`` keyword. Each knob left
    unset keeps the engine default."""

    def __init__(
        self,
        sweep_budget: int | None = None,
        active_set_budget: int | None = None,
        relaxation: float | None = None,
        tolerance: float | None = None,
    ) -> None: ...
    @property
    def sweep_budget(self) -> int: ...
    @property
    def active_set_budget(self) -> int | None: ...
    @property
    def relaxation(self) -> float: ...
    @property
    def tolerance(self) -> float: ...

def validate_model(model_json: str) -> None: ...
def simulate_json(
    model_json: str,
    t_max: float,
    journal: bool = False,
    confluence_check: bool = False,
    samples: list[float] | None = None,
    seed: int = 0,
    rng_stream: int = 0,
    flow: FlowConfig | None = None,
    max_transition_firings: int | None = None,
    max_flow_restarts: int | None = None,
) -> str: ...
def monte_carlo_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    samples: list[float],
    seed: int = 0,
    threads: int | None = None,
    quantiles: list[float] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_step: float | None = None,
    tol_event: float | None = None,
    sub_samples: int | None = None,
    stop_at_targets: bool = False,
    flow: FlowConfig | None = None,
) -> str: ...
def analyse_sequences_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
) -> str: ...

class Snapshot:
    """Opaque interactive-session checkpoint (see ``Interactive``)."""

class Interactive:
    """Low-level stateful step-by-step engine (JSON in/out); wrapped by
    the Pythonic ``pyraichu.Interactive``."""

    def __init__(
        self,
        model_json: str,
        t_max: float,
        journal: bool = False,
        confluence_check: bool = False,
        seed: int = 0,
        rng_stream: int = 0,
        flow: FlowConfig | None = None,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    def fireable(self) -> str: ...
    def attribute(self, qualified: str) -> str | None: ...
    def state(self, qualified: str) -> str | None: ...
    def history(self) -> str: ...
    def fire(self, name: str, to: str | None = None) -> str: ...
    def step(self) -> str | None: ...
    def set_date(self, name: str, date: float) -> None: ...
    def reset(self) -> None: ...
    def snapshot(self) -> Snapshot: ...
    def restore(self, snap: Snapshot) -> None: ...
