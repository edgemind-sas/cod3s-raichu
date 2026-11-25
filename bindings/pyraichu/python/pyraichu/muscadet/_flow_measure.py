"""Measurement links of a component: declaration and emission."""

from __future__ import annotations

from typing import Any

from ._expr import _float, _float_attribute, _var
from ._records import (
    _fill_alias,
    _level_alias,
    _MeasurementIn,
    _MeasurementOut,
    _ratio_alias,
)


class _MeasurementsMixin:
    """Methods of :class:`ObjFlow` measurement links of a component: declaration and emission."""

    def add_measurement_in(self, name: str, flows: list[str] | None = None) -> None:
        """Declare the reading side of a measurement link (muscadet
        `add_measurement_in`).

        The observer reads a published level and fill through ordinary in
        ports carrying no channel: the link is read-only by construction,
        exchanges no quantity and enters no allocation. `name` matches the
        observed capacity's name, which is what lines the two sides up;
        `flows` names constituents of the observed volume to read beside
        the totals, and a constituent the publisher does not hold is
        refused at connection.

        Connect with :meth:`System.connect_measurement`."""
        if any(existing.name == name for existing in self.measurements_in):
            raise ValueError(
                f"ObjFlow `{self.name}`: measurement link `{name}` is already declared"
            )
        if any(existing.name == name for existing in self.capacities):
            raise ValueError(
                f"ObjFlow `{self.name}`: measurement link `{name}` clashes with "
                "the capacity of that name"
            )
        self.measurements_in.append(_MeasurementIn(name=name, flows=list(flows or [])))

    def add_measurement_out(
        self,
        name: str,
        source: str | None = None,
        flows: list[str] | None = None,
        level_default: float = 0.0,
        fill_default: float = 0.0,
        ratio_default: float = 0.0,
        gain_default: float = 1.0,
    ) -> None:
        """Declare a measurement this component PUBLISHES (muscadet
        `add_measurement_out`, R37): what makes a redundant instrument set
        expressible, an instrument being a component that can fail on its
        own.

        `source` names a capacity or a measurement channel of this same
        component, whose level (and fill, and the constituents in `flows`
        with their shares) is republished at every evaluation point, times
        ``{name}_level_gain`` (created at `gain_default`): the endpoint a
        failure mode clamps, 0 for a dead instrument, 5 for a wild one.
        Without a source, ``{name}_level`` is a plain variable holding
        `level_default`. The published ports are a capacity's own, so an
        observer declares an ordinary `add_measurement_in` and is wired with
        :meth:`System.connect_measurement`.

        A continuous output's rate is not a source here: a flow publishes
        its own rate channel (`publish_rate`), which an observer reads
        directly."""
        where = f"ObjFlow `{self.name}`: published measurement `{name}`"
        if any(existing.name == name for existing in self.measurements_out):
            raise ValueError(f"{where} is already declared")
        if any(existing.name == name for existing in self.capacities):
            raise ValueError(
                f"{where} clashes with the capacity of that name, which "
                "publishes its level under the same aliases"
            )
        if any(existing.name == name for existing in self.measurements_in):
            raise ValueError(
                f"{where} clashes with the measurement link of that name, "
                "whose readings carry the same aliases"
            )
        flows = list(flows or [])
        if len(set(flows)) != len(flows):
            raise ValueError(f"{where} names a constituent more than once: {flows}")
        self.measurements_out.append(
            _MeasurementOut(
                name=name,
                source=source,
                flows=flows,
                level_default=float(level_default),
                fill_default=float(fill_default),
                ratio_default=float(ratio_default),
                gain_default=float(gain_default),
            )
        )

    def _measures(self, name: str | None) -> bool:
        """Whether `name` designates a measurement channel of this
        component.

        Every channel a link materialises reads a **published capacity
        level**: the level and the weighted fill of the observed volume,
        plus the same pair per constituent read
        (:meth:`_MeasurementIn.channels` against
        :meth:`_Capacity.published`). Each of them is an integrated
        content or a sweep over integrated contents, so a measurement is
        continuous whichever channel it names and the predicate needs no
        case distinction."""
        return any(name in link.channels() for link in self.measurements_in)

    def _build_measurements(
        self, variables: list[dict], ports: list[dict], equations: list[dict]
    ) -> None:
        """The reading side of each measurement link: one variable and one
        in port per published channel, each swept off its port.

        The ports carry no channel, so the link exchanges no quantity and
        the reader enters no allocation operator."""
        me = self.name
        for measurement in self.measurements_in:
            for variable in measurement.channels():
                port = f"{variable}_in"
                variables.append(_float_attribute(variable, 0.0))
                ports.append({"name": port, "dir": "in"})
                equations.append(
                    {
                        "target": variable,
                        "kind": "explicit",
                        "expr": {
                            "op": "port_agg",
                            "port": {"component": me, "port": port},
                            "agg": "sum",
                        },
                    }
                )

    def _measurement_out_readings(
        self, publication: _MeasurementOut
    ) -> dict[str, dict[str, Any]]:
        """What each alias of `publication` republishes, before the gain:
        read off its source, a capacity or a measurement channel of this
        component, under the aliases that source publishes itself.

        A share its source does not publish (a single-constituent volume,
        or a channel whose volume is one) is the constituent's level over
        the total, 0 on an empty source, which is what a volume publishes
        where it does publish one."""
        me = self.name
        where = f"ObjFlow `{me}`: published measurement `{publication.name}`"
        source = publication.source
        capacity = next((c for c in self.capacities if c.name == source), None)
        channel = next((m for m in self.measurements_in if m.name == source), None)
        if capacity is not None:
            held = capacity.flow_names
            values = dict(capacity.published())
            ratios = capacity.publishes_ratios
        elif channel is not None:
            held = channel.flows
            values = {alias: alias for alias in channel.channels()}
            ratios = channel.ratios
        else:
            if any(flow.name == source for flow in self.flows_continuous_out):
                raise ValueError(
                    f"{where} names the continuous output `{source}` as its "
                    "source. A flow publishes its own rate channel "
                    "(`publish_rate`), which an observer reads directly, so "
                    "republishing it through an instrument is not carried"
                )
            raise ValueError(
                f"{where} names no capacity or measurement channel of this "
                f"component as its source: `{source}`"
            )
        missing = [flow for flow in publication.flows if flow not in held]
        if missing:
            raise ValueError(
                f"{where} republishes the constituents {missing}, which its "
                f"source `{source}` does not carry (it carries {list(held)})"
            )

        def read(alias: str) -> dict[str, Any]:
            return _var(me, values[alias])

        readings = {
            _level_alias(publication.name): read(_level_alias(source)),
            _fill_alias(publication.name): read(_fill_alias(source)),
        }
        for flow in publication.flows:
            level = read(_level_alias(source, flow))
            readings[_level_alias(publication.name, flow)] = level
            readings[_fill_alias(publication.name, flow)] = read(
                _fill_alias(source, flow)
            )
            if ratios:
                share = read(_ratio_alias(source, flow))
            else:
                total = read(_level_alias(source))
                share = {
                    "op": "if",
                    "cond": {"op": "cmp", "cmp": "gt", "lhs": total, "rhs": _float(0.0)},
                    "then": {"op": "div", "lhs": level, "rhs": total},
                    "otherwise": _float(0.0),
                }
            readings[_ratio_alias(publication.name, flow)] = share
        return readings

    def _build_measurements_out(
        self, variables: list[dict], ports: list[dict], equations: list[dict]
    ) -> None:
        """The publishing side of each instrument: one variable per alias,
        each exported on the out port a capacity would use, each swept as
        its source's reading times the gain."""
        me = self.name
        for publication in self.measurements_out:
            defaults = {_level_alias(publication.name): publication.level_default,
                        _fill_alias(publication.name): publication.fill_default}
            for flow in publication.flows:
                defaults[_level_alias(publication.name, flow)] = publication.level_default
                defaults[_fill_alias(publication.name, flow)] = publication.fill_default
                defaults[_ratio_alias(publication.name, flow)] = publication.ratio_default
            for alias, attribute in publication.published():
                variables.append(_float_attribute(attribute, defaults[alias]))
                ports.append({"name": f"{alias}_out", "dir": "out", "attr": attribute})
            variables.append(_float_attribute(publication.gain, publication.gain_default))
            if publication.source is None:
                continue
            gain = _var(me, publication.gain)
            for alias, reading in self._measurement_out_readings(publication).items():
                equations.append(
                    {
                        "target": alias,
                        "kind": "explicit",
                        "expr": {"op": "mul", "args": [gain, reading]},
                    }
                )
