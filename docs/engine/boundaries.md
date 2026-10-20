# Boundaries

The tank now has a level. Between two events the engine integrates the
level's differential equation; a **watched** transition fires the instant
the level crosses one of its thresholds. This chapter shows how that
crossing takes its place among the dates of the
[race](fixpoint-and-race.md). The model is the second one of the
[introduction](index.md).

<div class="engine-player" data-trace="continuous-level" data-page="continuous"></div>

<script type="application/json" id="trace-continuous-level">
--8<-- "docs/engine/traces/continuous-level.json"
</script>

<script type="application/json" id="engine-player-text-continuous">
{
 "captions": {
  "stable": "Stable state at {time}: no function is pending. {armed} stochastic transition(s) are armed.",
  "integrate": "The engine integrates the level from {from} toward the next clock date, {next_date}. No threshold is crossed on the way, so the dated transition fires at {to}.",
  "integrate_threshold": "The engine integrates the level from {from} toward the next clock date, {next_date}. The level reaches a threshold first: integration stops at {to}, where the watched transition fires. The clock date stays in the race.",
  "race": "Next event: {winner}. The armed dates still waiting are kept.",
  "fired": "Event at {time}: {transition} fires, {from} -> {to}.",
  "iteration": "Iteration {index} of the fixpoint: {function} runs and writes {count} attribute(s)."
 },
 "going_further": {
  "integrate": {"rule": "evolC", "text": "Between events the continuous state follows its differential equation, integrated with an adaptive Dormand-Prince method up to the next scheduled date.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/continuous.rs"},
  "integrate_threshold": {"rule": "schWT", "text": "A watched transition has no sampled date. While integrating, the engine watches its guard and, when a scan of the step's dense output finds the guard flipped, locates the crossing by bisection. The crossing date is never predicted in advance.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/continuous.rs"},
  "race": {"rule": "schST, evolT", "text": "Dated transitions keep their sampled dates while the level evolves; only a change of state or rate reschedules them.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/schedule.rs"},
  "fired": {"rule": "evolT", "text": "A watched transition fires like any other: one automaton changes state, then the fixpoint runs.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/run.rs"},
  "iteration": {"rule": "evolA", "text": "The same fixpoint as in the discrete chapter: one function application per iteration, lowest index first.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/run.rs"},
  "stable": {"rule": "updateMT, updateIT", "text": "The schedule is updated after the fixpoint, exactly as in the discrete chapter.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/schedule.rs"}
 },
 "thresholds": [
  {"value": 8, "label": "HIGH"}, {"value": 7, "label": "HIGH_RESET"},
  {"value": 3, "label": "LOW_RESET"}, {"value": 2, "label": "LOW"}
 ],
 "labels": {
  "P1.mode.fail": "P1 fails", "P1.mode.repair": "P1 repaired",
  "P2.mode.start": "P2 start", "P2.mode.release": "P2 released",
  "P2.mode.fail": "P2 fails", "P2.mode.repair": "P2 repaired",
  "Tank.regime.reach_high": "level reaches HIGH", "Tank.regime.leave_high": "level back below HIGH_RESET",
  "Tank.regime.reach_low": "level reaches LOW", "Tank.regime.leave_low": "level back above LOW_RESET",
  "P1.update_P1": "update_P1", "P2.update_P2": "update_P2", "Tank.update_Tank": "update_Tank"
 }
}
</script>

## Reading the steps

**Integration toward the next date.** Between events the level follows its
equation: it rises at 1 m/h while `P1` delivers 2 m/h against a consumption
of 1 m/h. The engine integrates toward the earliest pending date of the race,
here the date sampled for `P1 fails`.

**Interrupted by a threshold.** The level reaches `HIGH` at 3 h, long before
that date. The transition `level reaches HIGH` has no date of its own: its
guard reads the level, so the engine watches it during integration and stops
where it flips, located to the event tolerance. The crossing is located, not predicted: the engine
never computes in advance when the level will get there.

**The race goes on.** The date of `P1 fails` is untouched by the crossing.
After the fixpoint (the main pump is throttled, so the level holds), the
engine integrates again toward the same date.

**A failure changes the slope.** When `P1` fails, the backup delivers only
0.5 m/h: the level starts falling. Its next crossing, of `HIGH_RESET` then of
`LOW`, is again found during integration. In alarm the consumers are cut off,
so the level holds at `LOW` until a pump refills the tank.

For how a crossing is detected and located inside one integration step, the
scan spacing that bounds how short an episode can be and still be seen, and the
tolerance that bounds its date, see the
[numerical tuning guide](../guides/numerical-tuning.md#detection-and-location).
