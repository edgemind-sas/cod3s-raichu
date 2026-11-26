"""Time profiles of a continuous output: declaration and the expression they emit."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ._expr import _float
from ._records import _Profile
from ._vocabulary import (
    _CONTINUITY_MESSAGE,
    _MASK64,
    _PROFILE_REQUIRED_KEYS,
    DRAWN_PROFILE,
    flow_noise_key,
    PROFILE_CLASS_KEYS,
    PROFILE_CLASSES,
    SINE_PROFILE,
    SQUARE_PROFILE,
)


class _ProfilesMixin:
    """Methods of :class:`ObjFlow` time profiles of a continuous output: declaration and the expression they emit."""

    # --- time profiles --------------------------------------------------

    def _parse_profile(
        self, flow: str, spec: Any, side: str = "out"
    ) -> _Profile | None:
        """One declared time profile, normalised, or refused where it was
        written.

        The shapes a mapping can carry are :data:`PROFILE_CLASSES`: a bare
        callable is refused because the continuity flag is the one thing
        this layer cannot work out for itself, and a mapping naming the
        callable shape offers a serialised form that cannot be serialised.
        Each shape is refused on what muscadet's constructor refuses, so a
        declaration both engines read is refused by both.

        `side` is which end of the component declared the curve: a demand
        that varies with the clock is as ordinary as a production that
        does, and a refusal naming an out-flow that the component has not
        got sends a modeller looking for the wrong declaration."""
        where = f"ObjFlow `{self.name}`: continuous {side}-flow `{flow}`"
        shapes = ", ".join(f"`{shape}`" for shape in PROFILE_CLASSES)
        if spec is None or isinstance(spec, _Profile):
            return spec
        if callable(spec):
            raise ValueError(
                f"{where} declares its time profile as a Python function; "
                f"{_CONTINUITY_MESSAGE}. Declare it as one of the shapes "
                f'{shapes}, as {{"cls": <shape>, ...}}'
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{where} declares {spec!r} as a time profile, which is not "
                f'one: a profile is {{"cls": <shape>, ...}}, the shape one of '
                f"{shapes}"
            )

        params = dict(spec)
        shape = params.pop("cls", None)
        if shape is None:
            raise ValueError(
                f"{where} declares a time profile with no `cls` key naming "
                f"its shape; the declarable shapes are {shapes}"
            )
        if shape not in PROFILE_CLASSES:
            raise ValueError(
                f"{where} declares the time profile shape `{shape}`; the "
                f"declarable shapes are {shapes}, every other profile "
                f"being a Python function. {_CONTINUITY_MESSAGE}"
            )
        vocabulary = PROFILE_CLASS_KEYS[shape]
        unknown = sorted(set(params) - set(vocabulary))
        if unknown:
            raise ValueError(
                f"{where} declares a `{shape}` time profile carrying unknown "
                f"keys {unknown} (expected {list(vocabulary)})"
            )
        missing = [key for key in _PROFILE_REQUIRED_KEYS[shape] if params.get(key) is None]
        if missing:
            raise ValueError(
                f"{where} declares a `{shape}` time profile without {missing}, "
                "which the shape cannot do without"
            )

        # Read FIRST, so every refusal below can name the curve at fault:
        # a component declaring one profile per flow is told which of them
        # it wrote wrong, where naming the flow alone leaves a modeller
        # counting mappings.
        label = params.pop("name", None)
        if label is not None and not isinstance(label, str):
            raise ValueError(
                f"{where} declares a time profile named {label!r}; `name` is "
                "the label muscadet carries on the curve for its own messages, "
                "so it is a string, or it is left out and the curve is named "
                "after its shape"
            )
        # muscadet's `Profile.__init__` does `name or type(self).__name__`,
        # and the default it writes out is therefore the class name. Mirror
        # it rather than carrying `None`: a read-back of what this layer
        # read says what muscadet's read-back said.
        label = label or shape
        named = f" `{label}`"
        if label == SINE_PROFILE:
            named = ""

        period = float(params.get("period", 2 * math.pi))
        if not period > 0.0:
            raise ValueError(
                f"{where} declares a time profile{named} of period {period:g}; "
                "the period is the duration of one cycle and must be strictly "
                "positive"
            )
        value_min = float(params.get("value_min", 0.0))
        if value_min < 0.0:
            raise ValueError(
                f"{where} declares a time profile{named} clamped at value_min="
                f"{value_min:g}. A profile SCALES production, so a negative "
                "factor would mean a negative quantity, which no balance here "
                "models: use value_min=0 to cut the negative half-cycle, or "
                "lift the whole curve with an offset"
            )
        value_max = params.get("value_max")
        value_max = (
            None if value_max is None or value_max == math.inf else float(value_max)
        )
        if value_max is not None and value_max < value_min:
            raise ValueError(
                f"{where} declares a time profile{named} whose clamps are the "
                f"wrong way round: value_min={value_min:g} exceeds "
                f"value_max={value_max:g}"
            )
        profile = _Profile(
            amplitude=float(params.get("amplitude", 1.0)),
            period=period,
            phase_shift=float(params.get("phase_shift", 0.0)),
            offset=float(params.get("offset", 0.0)),
            value_min=value_min,
            value_max=value_max,
            name=label,
            shape=shape,
            **self._profile_noise(where, named, flow, params),
        )
        if shape == SQUARE_PROFILE:
            return self._square_profile(where, named, profile, params)
        if shape == DRAWN_PROFILE:
            return replace(
                profile, points=self._drawn_points(where, named, period, params["points"])
            )
        return profile

    def _profile_noise(
        self, where: str, named: str, flow: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """The noise modifier of a declared profile, refused on what
        muscadet refuses. A noisy profile declared without a key gets the
        key of this flow (:func:`flow_noise_key`), as muscadet gives it."""
        amplitude = float(params.get("noise_amplitude", 0.0))
        if not (amplitude >= 0.0 and math.isfinite(amplitude)):
            raise ValueError(
                f"{where} declares a time profile{named} with noise_amplitude="
                f"{amplitude:g}; the standard deviation of a noise is a finite "
                "number, not negative"
            )
        time_scale = params.get("noise_time_scale")
        if amplitude > 0.0:
            if time_scale is None:
                raise ValueError(
                    f"{where} declares a time profile{named} with a noise of "
                    f"amplitude {amplitude:g} and no noise_time_scale, the "
                    "spacing of the noise nodes"
                )
            time_scale = float(time_scale)
            if not (time_scale > 0.0 and math.isfinite(time_scale)):
                raise ValueError(
                    f"{where} declares a time profile{named} with "
                    f"noise_time_scale={time_scale:g}; the spacing of the noise "
                    "nodes is strictly positive and finite"
                )
        elif time_scale is not None:
            raise ValueError(
                f"{where} declares a time profile{named} with noise_time_scale="
                f"{time_scale!r} without noise; give a positive "
                "noise_amplitude, or leave both out"
            )
        key = params.get("noise_key")
        if key is not None and (
            isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= _MASK64
        ):
            raise ValueError(
                f"{where} declares a time profile{named} with noise_key={key!r}; "
                "a flow key is an integer that fits in 64 unsigned bits"
            )
        if amplitude > 0.0 and key is None:
            key = flow_noise_key(self.name, flow)
        if not amplitude > 0.0:
            return {}
        return {
            "noise_amplitude": amplitude,
            "noise_time_scale": time_scale,
            "noise_key": key,
        }

    @staticmethod
    def _square_profile(
        where: str, named: str, profile: _Profile, params: dict[str, Any]
    ) -> _Profile:
        """A square wave's levels and durations, refused on what
        muscadet's `TrapezoidProfile` refuses."""
        low, high = float(params["low"]), float(params["high"])
        high_duration = float(params["high_duration"])
        ramp_duration = float(params["ramp_duration"])
        if not ramp_duration > 0.0:
            raise ValueError(
                f"{where} declares a square wave{named} with ramp_duration="
                f"{ramp_duration:g}. A zero ramp is a jump, which the solver "
                "would integrate across without stopping at it; give the edge "
                "a duration, however short"
            )
        if high_duration < 0.0:
            raise ValueError(
                f"{where} declares a square wave{named} with a negative "
                f"high_duration={high_duration:g}"
            )
        if 2.0 * ramp_duration + high_duration > profile.period:
            raise ValueError(
                f"{where} declares a square wave{named} whose plateau "
                f"({high_duration:g}) and two ramps ({ramp_duration:g} each) "
                f"last longer than the period ({profile.period:g})"
            )
        for label, level in (("low", low), ("high", high)):
            if level < 0.0:
                raise ValueError(
                    f"{where} declares a square wave{named} with {label}="
                    f"{level:g}; a profile factor scales a quantity and may not "
                    "be negative"
                )
        return replace(
            profile,
            low=low,
            high=high,
            high_duration=high_duration,
            ramp_duration=ramp_duration,
        )

    @staticmethod
    def _drawn_points(
        where: str, named: str, period: float, declared: Any
    ) -> list[list[float]]:
        """A drawn curve's points, refused on what muscadet's
        `PiecewiseLinearProfile` refuses."""
        try:
            pairs = [list(pair) for pair in declared]
        except TypeError:
            raise ValueError(
                f"{where} declares a drawn curve{named} whose points {declared!r} "
                "are not a list of [instant, factor] pairs"
            ) from None
        if len(pairs) < 2:
            raise ValueError(
                f"{where} declares a drawn curve{named} with {len(pairs)} "
                "point(s); at least two points are needed"
            )
        points: list[list[float]] = []
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError(
                    f"{where} declares a drawn curve{named} whose point {index} "
                    f"is {pair!r}, not an [instant, factor] pair"
                )
            instant, factor = float(pair[0]), float(pair[1])
            if not (math.isfinite(instant) and math.isfinite(factor)):
                raise ValueError(
                    f"{where} declares a drawn curve{named} whose point {index} "
                    f"is not finite: {pair!r}"
                )
            if factor < 0.0:
                raise ValueError(
                    f"{where} declares a drawn curve{named} whose point {index} "
                    f"has factor {factor:g}, and a profile factor may not be "
                    "negative"
                )
            if not instant < period:
                raise ValueError(
                    f"{where} declares a drawn curve{named} whose point {index} "
                    f"is at instant {instant:g}, not below the period "
                    f"({period:g}); the curve returns to its first point at the "
                    "period by itself"
                )
            if points and not instant > points[-1][0]:
                raise ValueError(
                    f"{where} declares a drawn curve{named} whose instants are "
                    f"not strictly increasing: point {index} at {instant:g} "
                    f"follows {points[-1][0]:g}"
                )
            points.append([instant, factor])
        if points[0][0] != 0.0:
            raise ValueError(
                f"{where} declares a drawn curve{named} whose first point is at "
                f"instant {points[0][0]:g}; it must be at instant 0"
            )
        return points

    def _profile_expr(self, profile: _Profile) -> dict[str, Any]:
        """The clamped profile, as an expression of simulation time: its
        base curve, plus its noise, clamped."""
        wave: dict[str, Any]
        if profile.shape == SINE_PROFILE:
            angle = {
                "op": "mul",
                "args": [
                    {
                        "op": "sub",
                        "lhs": {"op": "time"},
                        "rhs": _float(profile.phase_shift),
                    },
                    _float(2 * math.pi / profile.period),
                ],
            }
            wave = {
                "op": "add",
                "args": [
                    {
                        "op": "mul",
                        "args": [
                            _float(profile.amplitude),
                            {"op": "sin", "arg": angle},
                        ],
                    },
                    _float(profile.offset),
                ],
            }
        else:
            # The square wave and the drawn curve are one engine node: a
            # periodic piecewise-linear signal, evaluated by binary search
            # without an event per breakpoint and per period.
            wave = {
                "op": "piecewise_linear",
                "arg": {"op": "time"},
                "points": profile.drawn_points(),
                "period": profile.period,
                "phase_shift": profile.phase_shift,
            }
        if profile.has_noise:
            wave = {
                "op": "add",
                "args": [
                    wave,
                    {
                        "op": "smooth_noise",
                        "arg": {"op": "time"},
                        "amplitude": profile.noise_amplitude,
                        "time_scale": profile.noise_time_scale,
                        "key": profile.noise_key,
                    },
                ],
            }
        clamped: dict[str, Any] = {
            "op": "max",
            "args": [_float(profile.value_min), wave],
        }
        if profile.value_max is None:
            return clamped
        return {"op": "min", "args": [clamped, _float(profile.value_max)]}
