# Fixpoint and race of dates

This chapter follows the [installation](index.md) through its first events.
Step through it with the buttons (or the arrow keys once the player has
focus). Everything shown was recorded from the engine on the model of the
introduction.

<div class="engine-player" data-trace="discrete-backup-starts" data-page="discrete"></div>

<script type="application/json" id="trace-discrete-backup-starts">
--8<-- "docs/engine/traces/discrete-backup-starts.json"
</script>

<script type="application/json" id="trace-discrete-backup-fails">
--8<-- "docs/engine/traces/discrete-backup-fails.json"
</script>

<script type="application/json" id="engine-player-text-discrete">
{
 "captions": {
  "stable": "Stable state at {time}: no function is pending. {armed} transition(s) are armed and waiting for their date.",
  "race": "Race of dates: {runners} armed transition(s), each with the date sampled when it was armed. The earliest, {winner} at {date}, fires next.",
  "fired": "Event at {time}: {transition} fires, {from} -> {to}. The functions that read this state are now pending.",
  "iteration": "Iteration {index} of the fixpoint: {function} runs and writes {count} attribute(s). What those attributes feed becomes pending.",
  "branch": "{transition} fires now, and its outcome is random. Pick one to see what the engine does with it."
 },
 "going_further": {
  "stable": {"rule": "updateMT, updateIT", "text": "After the fixpoint the engine reschedules the armed transitions whose rate changed and drops those whose guard turned false. A transition still armed with the same law keeps the date it already had.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/schedule.rs"},
  "race": {"rule": "schST, evolT", "text": "A stochastic transition gets its firing date from its law when it is armed. The engine always fires the earliest pending date: there is no random choice among transitions.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/schedule.rs"},
  "fired": {"rule": "evolT", "text": "Firing changes the state of one automaton and applies the transition's own edge effects, if it has any, before the fixpoint starts.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/run.rs"},
  "iteration": {"rule": "evolA", "text": "The engine keeps a set of pending sensitive functions and always applies the one with the lowest index. Each application is one iteration; the fixpoint ends when the set is empty.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/run.rs"},
  "branch": {"rule": "evolT (instantaneous transition)", "text": "An instantaneous transition fires as soon as its guard holds; its law draws which target it reaches. Here the draw is replaced by your choice.", "code": "https://github.com/edgemind-sas/cod3s-raichu/blob/main/crates/raichu-core/src/engine/run.rs"}
 },
 "labels": {
  "P1.mode.fail": "P1 fails", "P1.mode.repair": "P1 repaired",
  "P2.mode.start": "P2 start", "P2.mode.release": "P2 released",
  "P2.mode.fail": "P2 fails", "P2.mode.repair": "P2 repaired",
  "P1.update_P1": "update_P1", "P2.update_P2": "update_P2", "Tank.update_Tank": "update_Tank"
 }
}
</script>

## Reading the steps

**Stable state.** Nothing is pending: every attribute agrees with what its
function computes. The engine now only waits for the next date.

**Race of dates.** Each armed stochastic transition carries a date, sampled
from its law at the moment it was armed. The engine does not draw *which*
transition fires: it fires the one whose date comes first. At the start only
`P1 fails` is armed; after the backup starts, `P1 repaired` and `P2 fails`
race each other.

**Event.** One automaton changes state. Every sensitive function that reads
that state becomes pending.

**Iteration of the fixpoint.** The engine applies the pending functions one
at a time, always the one with the lowest index first. One iteration is one
function application: it recomputes all the attributes of its component, and
every function reading an attribute it changed becomes pending in turn. When
`P1` fails, three iterations follow: `update_P1` sets `P1.delivers` to false,
which makes `update_P2` and `update_Tank` pending; `update_P2` runs first
(lower index) and solicits the backup, then `update_Tank` finds the tank no
longer fed. An iteration may also write nothing, when the function recomputes
the values it already had.

**Back to stable.** When no function is pending, the engine updates the
schedule: a transition that became armed gets a date, one whose rate changed
gets a new date, one whose guard turned false is dropped.

**A random outcome.** The backup's start is an *instantaneous* transition: it
fires as soon as `P2` is solicited, and its law decides whether it reaches
`running` or `failed`. The player lets you pick either outcome and shows the
cascade each one triggers.

!!! note "What this scenario avoids"
    Two transitions firing at the very same date, and a fixpoint whose
    result would depend on the order of the functions, are real cases the
    engine handles (and diagnoses for the second). They are left out here
    so that each step shows one thing.
