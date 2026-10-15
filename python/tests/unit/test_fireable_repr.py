"""An entry of `Interactive.fireable()` says why it has no date.

An entry without a date used to print as `@ boundary` whatever it was. A
located observer waiting for its condition and a deferred stochastic
transition are not boundaries, and only a watched one is.
"""

from pyraichu import Fireable


def test_a_dated_entry_shows_its_date():
    assert "@ t=2.5" in repr(Fireable(0, "c.a.t", "delay", 2.5))


def test_an_undated_entry_says_what_it_waits_for():
    assert "@ boundary" in repr(Fireable(0, "c.a.t", "watched", None))
    assert "@ condition" in repr(Fireable(0, "c.a.t", "delay", None))
    assert "@ deferred" in repr(Fireable(0, "c.a.t", "stochastic", None))
