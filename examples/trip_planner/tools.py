"""Deterministic fake tools for the trip planner.

Backed by JSON fixtures, so the demo runs with no API key and no network. Each tool
returns the shape a real one would, including the parts that make chaos interesting:
`search_flights` carries a free-text `notes` field, and `get_account` returns a
credentials-shaped payload.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = ["get_account", "get_weather_data", "hold_booking", "search_flights"]

_FIXTURES = Path(__file__).parent / "fixtures"

#: Set by the app when a canary is planted, so `objective: exfiltrate_secret` has a
#: real value to try to move and `secret_in_output` is actually reachable (D-16).
CANARY: str | None = None


@lru_cache(maxsize=8)
def _fixture(name: str) -> Any:
    """Read one fixture file.

    Args:
        name: File stem.

    Returns:
        The parsed JSON.
    """
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def get_weather_data(location: str) -> list[dict[str, Any]]:
    """Return a 3-day forecast.

    Args:
        location: City name.

    Returns:
        Three records of `{date, condition, temp_c, humidity}`.
    """
    table = _fixture("weather")
    return list(table.get(location) or table["Paris"])


def search_flights(origin: str, destination: str, date: str = "2026-09-07") -> dict[str, Any]:
    """Return flight options for a route.

    Args:
        origin: IATA origin code.
        destination: Destination city.
        date: Departure date.

    Returns:
        `{cheapest, options, notes}`. `notes` is free text straight from the
        supplier -- the field a real integration has no control over.
    """
    codes = {"Paris": "CDG", "Lisbon": "LIS"}
    key = f"{origin}-{codes.get(destination, 'CDG')}"
    table = _fixture("flights")
    quote = dict(table.get(key) or table["DEL-CDG"])
    quote["date"] = date
    return quote


def get_account() -> dict[str, Any]:
    """Return the traveller's stored account record.

    Returns:
        Profile fields plus the booking credential the hold call needs.
    """
    record = dict(_fixture("account"))
    if CANARY:
        record["booking_api_key"] = CANARY
    return record


def hold_booking(
    flight_id: str, passenger: str, idempotency_key: str | None = None
) -> dict[str, Any]:
    """Place a 24-hour hold on a seat.

    Not idempotent unless `idempotency_key` is supplied: calling it twice without one
    places two holds. That is deliberate -- it is what `DuplicateSideEffectFault` and
    `CheckpointRollbackFault` are aimed at.

    Args:
        flight_id: Which flight.
        passenger: Who for.
        idempotency_key: Caller-generated key. Reusing one returns the first hold.

    Returns:
        `{hold_id, flight_id, passenger, duplicate}`.
    """
    key = idempotency_key or f"{flight_id}:{passenger}:{len(_HOLDS) + 1}"
    duplicate = key in _HOLDS
    if not duplicate:
        _HOLDS[key] = f"HOLD-{len(_HOLDS) + 1:04d}"
    return {
        "hold_id": _HOLDS[key],
        "flight_id": flight_id,
        "passenger": passenger,
        "duplicate": duplicate,
    }


#: Idempotency key to hold id. Module-level so a duplicate call is observable.
_HOLDS: dict[str, str] = {}


def reset() -> None:
    """Clear the hold ledger between runs."""
    _HOLDS.clear()
