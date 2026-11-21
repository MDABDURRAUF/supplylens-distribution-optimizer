"""Shipment consolidation: many small shipments against fewer larger ones.

A customer that receives several shipments a week pays a minimum charge and
a light-weight rate on each. Holding orders for a few days and sending them
together means fewer, heavier shipments at a lower rate per pound, and
sometimes a cheaper mode. The price is waiting time.

The rule used here: holding orders for up to ``h`` working days means
shipping at most once every ``h + 1`` working days, so a five-day week has
at most ``ceil(5 / (h + 1))`` shipments. A lane is consolidated only if
days held plus transit days still meet the service rule, which is read here
as the longest acceptable time from order to delivery.

Only freight changes. Handling, inventory, dock space and what the customer
thinks of fewer deliveries are not costed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import Network
from .optimizer import Result, Settings, _Model
from .rates import MODES

WORKING_DAYS = 5


def shipments_after_hold(shipments_per_week, hold_days: int) -> np.ndarray:
    """Shipments per week once orders are held up to ``hold_days`` working days (never more than today)."""
    if hold_days < 0 or int(hold_days) != hold_days:
        raise ValueError("hold_days must be a whole number of at least 0")
    return np.minimum(np.asarray(shipments_per_week, dtype=float), math.ceil(WORKING_DAYS / (int(hold_days) + 1)))


def consolidation_analysis(network: Network, plan: Result, settings: Settings = Settings(), max_hold_days: int = 2) -> pd.DataFrame:
    """For every lane of a plan, the cheapest way to ship if orders may be held up to ``max_hold_days``.

    One row per lane with the customer's ``shipments_per_week`` and the
    lane's ``mode`` today, the consolidated ones, pounds per shipment
    (averaged over the weeks), the ``hold_days`` chosen (0 = leave as is),
    ``order_to_delivery_days`` (days held plus transit days), freight before
    and after over all weeks, and ``saving``. The mode may change when
    heavier shipments make another mode cheaper. A hold is used only if it
    lowers freight and keeps days held plus transit days within
    ``settings.max_transit_days``. Largest saving first.
    """
    if not plan.ok:
        raise ValueError("the plan has not been costed; check its status and message")
    if max_hold_days < 0 or int(max_hold_days) != max_hold_days:
        raise ValueError("max_hold_days must be a whole number of at least 0")
    m = _Model(network, settings)
    lanes = plan.lanes.reset_index(drop=True)
    i = lanes["dc"].map({d: k for k, d in enumerate(m.dc_names)}).to_numpy(dtype=int)
    c = lanes["customer_id"].map({d: k for k, d in enumerate(m.ids)}).to_numpy(dtype=int)
    share, today = lanes["share"].to_numpy(dtype=float), m.per_week[c]

    best = lanes["freight_cost"].to_numpy(dtype=float).copy()
    hold = np.zeros(len(lanes), dtype=int)
    per_week = today.copy()
    mode = lanes["mode"].to_numpy(dtype=object).copy()
    days = lanes["transit_days"].to_numpy(dtype=int).copy()
    for h in range(1, int(max_hold_days) + 1):
        n = shipments_after_hold(today, h)
        pick, cost, transit, meets = m.best_modes(i, c, per_week=n, extra_days=h)
        cost = share * cost
        better = meets & (n < today) & (cost < best - 1e-6)
        best[better], hold[better], per_week[better], days[better] = cost[better], h, n[better], transit[better]
        mode[better] = np.array(MODES, dtype=object)[pick[better]]

    weeks = max(m.weeks, 1)
    weight = lanes["weight_lbs"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "customer_id": lanes["customer_id"], "city": lanes["city"], "state": lanes["state"], "dc": lanes["dc"], "share": share,
        "shipments_per_week": today.astype(int), "consolidated_shipments_per_week": per_week.astype(int), "hold_days": hold,
        "mode": lanes["mode"], "consolidated_mode": mode, "transit_days": days, "order_to_delivery_days": hold + days,
        "lbs_per_shipment": weight / (share * today * weeks), "consolidated_lbs_per_shipment": weight / (share * per_week * weeks),
        "freight_before": lanes["freight_cost"], "freight_after": best})
    out["saving"] = out["freight_before"] - out["freight_after"]
    return out.sort_values(["saving", "customer_id"], ascending=[False, True]).reset_index(drop=True)


def consolidation_summary(table: pd.DataFrame) -> dict:
    """Totals of a ``consolidation_analysis`` table.

    ``shipments_per_week_*`` count scheduled shipments (one per customer
    visit), not parcels or trucks; a split customer's shipments are counted
    once, by share. ``saving`` is an estimated freight saving
    over the weeks of demand, before any cost of holding goods.
    """
    before, after = float(table["freight_before"].sum()), float(table["freight_after"].sum())
    return {
        "lanes": int(len(table)),
        "lanes_consolidated": int((table["hold_days"] > 0).sum()),
        "mode_changes": int((table["mode"] != table["consolidated_mode"]).sum()),
        "shipments_per_week_before": int(round((table["share"] * table["shipments_per_week"]).sum())),
        "shipments_per_week_after": int(round((table["share"] * table["consolidated_shipments_per_week"]).sum())),
        "freight_before": before, "freight_after": after, "saving": before - after,
        "saving_pct": (before - after) / before if before else 0.0,
    }


def consolidation_by_window(network: Network, plan: Result, settings: Settings = Settings(), windows=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Freight saving from consolidation for each holding window: one row per ``max_hold_days``."""
    rows = []
    for window in windows:
        summary = consolidation_summary(consolidation_analysis(network, plan, settings, int(window)))
        rows.append({"max_hold_days": int(window), **{k: summary[k] for k in (
            "lanes_consolidated", "shipments_per_week_after", "freight_after", "saving", "saving_pct")}})
    return pd.DataFrame(rows)
