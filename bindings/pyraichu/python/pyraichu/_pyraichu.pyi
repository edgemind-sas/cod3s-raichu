"""Type stubs for the Rust extension module ``pyraichu._pyraichu``."""

__version__: str

#: Confidence level applied when a study states none (0.95).
DEFAULT_CONFIDENCE: float

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
def switching_loops_json(model_json: str) -> str: ...
def unfed_triggers_json(model_json: str) -> str: ...
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
    confidence: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_step: float | None = None,
    tol_event: float | None = None,
    sub_samples: int | None = None,
    stop_at_targets: bool = False,
    flow: FlowConfig | None = None,
    event_resolution: float | None = None,
) -> str: ...
def analyse_sequences_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
) -> str: ...
def run_sequences_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
    raw_path: str | None = None,
    observations: list[tuple[str, str, str, float]] | None = None,
    condition: tuple[str, str, float] | None = None,
) -> str: ...
def analyse_raw_sequences_json(
    raw_path: str, condition: tuple[str, str, float] | None = None
) -> str: ...
def importance_json(
    model_json: str,
    nb_runs: int,
    t_max: float,
    instants: list[float],
    target: str | None = None,
    seed: int = 0,
    threads: int | None = None,
    flow: FlowConfig | None = None,
    options_json: str | None = None,
) -> str: ...
def explore_json(
    model_json: str,
    target: str,
    horizon: float,
    algorithm: str = "exact",
    min_probability: float | None = None,
    max_length: int | None = None,
    max_failures: int | None = None,
    max_branches: int | None = None,
    gap_tolerance: float | None = None,
    rel_precision: float | None = None,
    max_terms: int | None = None,
    threads: int | None = None,
    level: int | None = None,
    refine: bool = True,
) -> str: ...
def exploration_domain_json(model_json: str) -> str: ...
def fault_tree_json(
    model_json: str,
    top_json: str | None = None,
    profile_json: str | None = None,
    max_nodes: int | None = None,
    cut_set_limit: int = 100_000,
    name: str = "fault_tree",
    targets_json: str | None = None,
    cut_sets: bool = True,
) -> str: ...
def fault_tree_envelope_json(
    model_json: str,
    mission_times: list[float],
    top_json: str | None = None,
    targets_json: str | None = None,
    profile_json: str | None = None,
    max_nodes: int | None = None,
    name: str = "fault_tree",
    max_bdd_nodes: int = 10_000_000,
    cut_set_limit: int = 100_000,
    cut_sets: bool = True,
    engine: str = "auto",
    max_order: int | None = None,
    min_cut_probability: float = 0.0,
    max_cut_sets: int = 1_000_000,
    max_expansions: int = 100_000_000,
) -> str: ...
def validate_fault_tree_structure(structure_json: str) -> None: ...
def validate_fault_tree_envelope(envelope_json: str) -> None: ...
def fault_tree_quantify_json(
    open_psa: str,
    top: str | None = None,
    mission_time: float | None = None,
    max_bdd_nodes: int = 10_000_000,
    cut_set_limit: int = 100_000,
    cut_sets: bool = True,
    engine: str = "auto",
    max_order: int | None = None,
    min_cut_probability: float = 0.0,
    max_cut_sets: int = 1_000_000,
    max_expansions: int = 100_000_000,
) -> str: ...
def validate_exploration(result_json: str) -> None: ...
def quantify_json(
    model_json: str,
    study_json: str,
    method: str,
    settings_json: str | None = None,
) -> str: ...
def validate_quantification(envelope_json: str) -> None: ...
def exploration_minimal_sequences_json(result_json: str) -> str: ...

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
        allow_fmu_import: bool = False,
        fmu_base_dir: str | None = None,
        operator_control: bool = False,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    def fireable(self) -> str: ...
    def attribute(self, qualified: str) -> str | None: ...
    def state(self, qualified: str) -> str | None: ...
    def history(self) -> str: ...
    def fire(self, name: str, to: str | None = None) -> str: ...
    def step(self) -> str | None: ...
    def advance_to(self, date: float) -> None: ...
    def advance_operator_to(self, date: float, max_events: int = 10_000) -> str: ...
    def set_date(self, name: str, date: float) -> None: ...
    def reset(self) -> None: ...
    def snapshot(self) -> Snapshot: ...
    def restore(self, snap: Snapshot) -> None: ...
