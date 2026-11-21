"""Freight rates and transit times for three modes: parcel, LTL and truckload.

These rate functions are illustrative. They have the shape of real freight
pricing (a minimum charge, cost rising with distance and weight, cheaper per
pound as a shipment gets heavier, a fuel surcharge) but the numbers are made
up and are not any carrier's tariff. Replace the ``RateCard`` values with
your own contract rates before trusting a cost.

One shipment of ``w`` pounds over ``d`` road miles costs, before fuel:

parcel      ``max(parcel_min, parcel_base + w * (parcel_per_lb + parcel_per_lb_mile * d))``
            per package, with packages of at most ``parcel_max_lbs``.
LTL         ``max(ltl_min, w / 100 * (ltl_base_cwt + ltl_cwt_per_mile * d) * discount)``
            where the discount depends on the weight break the shipment falls
            in. A shipment is billed as if it weighed the start of a heavier
            break whenever that is cheaper, so the charge never falls as
            weight rises. At most ``ltl_max_lbs`` per shipment.
truckload   ``max(tl_min, tl_per_mile * d)`` per truck, whatever the weight,
            with at most ``tl_max_lbs`` per truck.

A shipment heavier than the mode's limit is sent as the smallest number of
equal shipments that fit. The fuel surcharge is a percentage added to the
whole charge of every mode.

Transit days are ``handling days + miles / miles per day``, rounded up, and
never less than one day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("parcel", "LTL", "truckload")


@dataclass(frozen=True)
class RateCard:
    """Rate and transit-time parameters. Money in dollars, weight in pounds, distance in road miles."""

    fuel_surcharge: float = 0.20          # added to every freight charge, e.g. 0.20 = 20%

    parcel_base: float = 9.00             # per package
    parcel_per_lb: float = 0.28
    parcel_per_lb_mile: float = 0.00045
    parcel_min: float = 12.00
    parcel_max_lbs: float = 150.0
    parcel_miles_per_day: float = 400.0
    parcel_handling_days: float = 0.0

    ltl_base_cwt: float = 14.00           # per 100 lb
    ltl_cwt_per_mile: float = 0.035       # per 100 lb per mile
    # Weight breaks: (from this weight in lb, multiplier on the per-100-lb rate).
    ltl_breaks: tuple = ((0, 1.00), (500, 0.88), (1000, 0.76), (2000, 0.64), (5000, 0.54), (10000, 0.46))
    ltl_min: float = 110.00
    ltl_max_lbs: float = 20000.0
    ltl_miles_per_day: float = 450.0
    ltl_handling_days: float = 1.0        # pickup and terminal handling

    tl_per_mile: float = 2.30             # per truck
    tl_min: float = 650.00
    tl_max_lbs: float = 42000.0
    tl_miles_per_day: float = 500.0
    tl_handling_days: float = 0.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if name == "ltl_breaks":
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number of at least 0")
        for name in ("parcel_max_lbs", "ltl_max_lbs", "tl_max_lbs", "parcel_miles_per_day", "ltl_miles_per_day", "tl_miles_per_day"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        breaks = tuple((float(a), float(b)) for a, b in self.ltl_breaks)
        starts = [a for a, _ in breaks]
        if not breaks or starts[0] != 0 or starts != sorted(set(starts)) or any(b <= 0 for _, b in breaks):
            raise ValueError("ltl_breaks must start at weight 0, rise in weight, and have positive multipliers")
        object.__setattr__(self, "ltl_breaks", breaks)


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"'{mode}' is not a known mode; use one of: {', '.join(MODES)}")
    return mode


def max_weight(mode: str, rates: RateCard = RateCard()) -> float:
    """Heaviest single shipment the mode takes, in pounds."""
    return {"parcel": rates.parcel_max_lbs, "LTL": rates.ltl_max_lbs, "truckload": rates.tl_max_lbs}[_check_mode(mode)]


def _one_shipment(mode: str, miles: np.ndarray, weight: np.ndarray, rates: RateCard) -> np.ndarray:
    """Charge before fuel for one shipment that is within the mode's weight limit."""
    if mode == "parcel":
        return np.maximum(rates.parcel_min, rates.parcel_base + weight * (rates.parcel_per_lb + rates.parcel_per_lb_mile * miles))
    if mode == "LTL":
        per_cwt = rates.ltl_base_cwt + rates.ltl_cwt_per_mile * miles
        best = np.full(np.shape(weight), np.inf)
        for start, multiplier in rates.ltl_breaks:
            # Billed at this break: the actual weight, or the break's starting weight if the shipment is lighter.
            best = np.minimum(best, np.maximum(weight, start) / 100.0 * per_cwt * multiplier)
        return np.maximum(rates.ltl_min, best)
    return np.maximum(rates.tl_min, rates.tl_per_mile * miles) + 0.0 * weight


def shipment_cost(mode: str, miles, weight_lbs, rates: RateCard = RateCard()) -> np.ndarray:
    """Freight charge in dollars, fuel surcharge included, for shipping ``weight_lbs`` over ``miles``.

    Arrays are allowed and broadcast together. A weight above the mode's
    limit is split into the fewest equal shipments that fit, each charged
    separately. Zero weight costs nothing.
    """
    _check_mode(mode)
    miles, weight = np.broadcast_arrays(np.asarray(miles, dtype=float), np.asarray(weight_lbs, dtype=float))
    if (miles < 0).any() or (weight < 0).any():
        raise ValueError("miles and weight_lbs cannot be negative")
    pieces = np.ceil(np.round(weight / max_weight(mode, rates), 9))
    each = weight / np.maximum(pieces, 1.0)
    return pieces * _one_shipment(mode, miles, each, rates) * (1.0 + rates.fuel_surcharge)


def shipment_count(mode: str, weight_lbs, rates: RateCard = RateCard()) -> np.ndarray:
    """Number of packages, LTL shipments or trucks needed for ``weight_lbs``."""
    return np.ceil(np.round(np.asarray(weight_lbs, dtype=float) / max_weight(mode, rates), 9))


def transit_days(mode: str, miles, rates: RateCard = RateCard()) -> np.ndarray:
    """Whole days from leaving the distribution center to delivery, at least one."""
    _check_mode(mode)
    speed, handling = {"parcel": (rates.parcel_miles_per_day, rates.parcel_handling_days),
                       "LTL": (rates.ltl_miles_per_day, rates.ltl_handling_days),
                       "truckload": (rates.tl_miles_per_day, rates.tl_handling_days)}[mode]
    days = np.ceil(np.round(handling + np.asarray(miles, dtype=float) / speed, 9))
    return np.maximum(1, days).astype(int)


def rate_curve(miles: float, weights, rates: RateCard = RateCard()) -> dict:
    """Charge for one shipment at each weight, by mode, over a fixed distance: ``{mode: array}``."""
    return {mode: shipment_cost(mode, miles, weights, rates) for mode in MODES}
