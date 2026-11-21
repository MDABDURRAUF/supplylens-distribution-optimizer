"""What-if tests: fuel surcharge, demand growth, loss of a DC, and which DCs to operate.

Each test re-runs the optimizer under changed conditions and compares the
answer with the plan for today's conditions. All differences are planning
estimates under the stated rates and demand, not realized savings or losses.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .data import Network
from .optimizer import Result, Settings, baseline_plan, cost_plan, optimize
from .rates import MODES


def _moved(a: Result, b: Result) -> int:
    """Number of customers whose set of DCs differs between two plans."""
    first = a.lanes.groupby("customer_id")["dc"].agg(lambda s: tuple(sorted(s)))
    second = b.lanes.groupby("customer_id")["dc"].agg(lambda s: tuple(sorted(s)))
    return int((first != second.reindex(first.index)).sum())


def fuel_sensitivity(network: Network, settings: Settings = Settings(), surcharges=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """Cost of three plans at each fuel surcharge.

    ``baseline_cost``       today's allocation
    ``keep_plan_cost``      the plan optimized at the current surcharge, left unchanged
    ``reoptimized_cost``    a plan optimized again at that surcharge
    ``saving_vs_baseline``  baseline minus reoptimized
    ``reoptimizing_gain``   keep-plan minus reoptimized: what re-planning for the new fuel price adds
    ``customers_moved``     customers whose DC differs from the current plan

    The surcharge raises every freight charge by the same percentage, so it
    changes the plan only through the balance between freight and the costs
    that fuel does not touch (handling and fixed DC cost).
    """
    current = optimize(network, settings)
    if not current.ok:
        raise ValueError(f"No plan under today's conditions ({current.message}); fix that before testing fuel")
    rows = []
    for fuel in surcharges:
        at = settings.with_fuel(fuel)
        base = baseline_plan(network, at)
        keep = cost_plan(network, current.assignment, at, open_dcs=current.open_dcs)
        again = optimize(network, at)
        ok = again.ok
        rows.append({"fuel_surcharge": float(fuel), "baseline_cost": base.total_cost, "keep_plan_cost": keep.total_cost,
                     "reoptimized_cost": again.total_cost if ok else np.nan,
                     "saving_vs_baseline": base.total_cost - again.total_cost if ok else np.nan,
                     "reoptimizing_gain": keep.total_cost - again.total_cost if ok else np.nan,
                     "customers_moved": _moved(current, again) if ok else 0,
                     "dcs_open": len(again.open_dcs) if ok else 0})
    return pd.DataFrame(rows)


def demand_growth(network: Network, settings: Settings = Settings(), growths=(-0.10, 0.0, 0.10, 0.20, 0.30, 0.40)) -> pd.DataFrame:
    """Re-plan with all demand changed by each growth rate (0.10 = 10% more units on every line).

    One row per rate with the plan's cost, cost per unit, the fullest DC's
    use of capacity in its busiest week, how many DCs are full, and how many
    customers had to be split. Where no plan exists the status is
    ``infeasible`` and ``note`` says why and which rule to relax.
    """
    rows = []
    for growth in growths:
        grown = network.with_demand_growth(float(growth))
        plan = optimize(grown, settings)
        row = {"growth": float(growth), "status": plan.status, "units": float(grown.demand["units"].sum())}
        if plan.ok:
            used = plan.dc_summary[plan.dc_summary["open"]]
            row.update(total_cost=plan.total_cost, cost_per_unit=plan.total_cost / plan.units, freight_cost=plan.freight_cost,
                       handling_cost=plan.handling_cost, fixed_cost=plan.fixed_cost,
                       peak_utilization=float(used["peak_utilization"].max()),
                       dcs_full=int((used["peak_utilization"] >= 0.999).sum()), split_customers=plan.split_customers,
                       dcs_open=len(plan.open_dcs), note="")
        else:
            fixes = diagnose(grown, settings)
            row.update(note=plan.message + (f" Relaxing {_join(fixes)} would make a plan possible." if fixes else ""))
        rows.append(row)
    columns = ["growth", "status", "units", "total_cost", "cost_per_unit", "freight_cost", "handling_cost", "fixed_cost",
               "peak_utilization", "dcs_full", "split_customers", "dcs_open", "note"]
    return pd.DataFrame(rows).reindex(columns=columns)


def dc_loss(network: Network, settings: Settings, dc: str, current: Result | None = None) -> dict:
    """Effect of losing one DC entirely: the best plan without it.

    Returns ``current`` (today's optimized plan), ``adapt`` (the best plan
    without the DC, which may be infeasible), the customers and units the
    current plan ships from the DC, ``fixes`` (rules to relax if no plan
    exists) and ``extra_cost``: the change in total cost, leaving out the
    lost DC's own fixed cost on both sides, because whether that is still
    paid depends on why the DC is lost. NaN when no plan is possible.
    """
    current = optimize(network, settings) if current is None else current
    remaining = network.without_dc(dc)
    adapt = optimize(remaining, settings)
    lanes = current.lanes[current.lanes["dc"] == dc] if current.ok else current.lanes
    own_fixed = float(current.dc_summary.loc[current.dc_summary["dc"] == dc, "fixed_cost"].sum()) if current.ok else 0.0
    ok = current.ok and adapt.ok
    return {"current": current, "adapt": adapt, "customers_affected": int(lanes["customer_id"].nunique()),
            "units_affected": float(lanes["units"].sum()),
            "extra_cost": adapt.total_cost - (current.total_cost - own_fixed) if ok else float("nan"),
            "fixes": [] if adapt.ok else diagnose(remaining, settings)}


def dc_loss_table(network: Network, settings: Settings = Settings()) -> pd.DataFrame:
    """Lose each DC in turn and re-plan: one row per DC, worst first.

    ``outcome`` is ``absorbed`` when the remaining DCs can still serve every
    customer within the rules, else ``no feasible plan`` with the rules to
    relax in ``relax``. ``status`` is the solver's status for the new plan
    (``feasible`` means the time limit was reached, so the extra cost may be
    a little overstated). ``extra_cost`` is defined as in ``dc_loss``.
    """
    current = optimize(network, settings)
    if not current.ok:
        raise ValueError(f"No plan under today's conditions ({current.message}); fix that before testing DC loss")
    base = current.total_cost
    rows = []
    for dc in network.dcs["dc"]:
        out = dc_loss(network, settings, dc, current)
        adapt = out["adapt"]
        rows.append({"dc_lost": dc, "customers_affected": out["customers_affected"], "units_affected": out["units_affected"],
                     "outcome": "absorbed" if adapt.ok else "no feasible plan", "status": adapt.status, "extra_cost": out["extra_cost"],
                     "extra_cost_pct": out["extra_cost"] / base if base else np.nan,
                     "share_within_service": adapt.share_within_service if adapt.ok else np.nan,
                     "average_transit_days": adapt.average_transit_days if adapt.ok else np.nan,
                     "relax": _join(out["fixes"]) if not adapt.ok else ""})
    table = pd.DataFrame(rows)
    table["_rank"] = table["extra_cost"].fillna(np.inf)
    return table.sort_values("_rank", ascending=False).drop(columns="_rank").reset_index(drop=True)


def dc_count_table(network: Network, settings: Settings = Settings()) -> pd.DataFrame:
    """The best network with exactly 1, 2, ... DCs open: which ones, and what it costs in total.

    Shows the trade between fixed cost (fewer DCs) and freight (more DCs,
    closer to customers). Rows with no feasible plan have status
    ``infeasible`` and empty costs.
    """
    rows = []
    for k in range(1, len(network.dcs) + 1):
        plan = optimize(network, settings, n_open=k)
        row = {"dcs_open": k, "status": plan.status, "open": ", ".join(plan.open_dcs)}
        if plan.ok:
            row.update(total_cost=plan.total_cost, freight_cost=plan.freight_cost, handling_cost=plan.handling_cost,
                       fixed_cost=plan.fixed_cost, share_within_service=plan.share_within_service,
                       average_transit_days=plan.average_transit_days)
        rows.append(row)
    columns = ["dcs_open", "status", "open", "total_cost", "freight_cost", "handling_cost", "fixed_cost", "share_within_service",
               "average_transit_days"]
    return pd.DataFrame(rows).reindex(columns=columns)


def open_close(network: Network, settings: Settings = Settings()) -> dict:
    """Which DCs to operate: the plan with every DC open against the plan free to close DCs.

    Returns ``all_open``, ``chosen``, the DCs ``closed``, and ``saving``
    (total cost with all open minus total cost of the chosen network; NaN if
    either has no plan). Closing costs, lease terms and the risk of running
    fewer sites are not in the model.
    """
    all_open = optimize(network, replace(settings, choose_dcs=False))
    chosen = optimize(network, replace(settings, choose_dcs=True))
    ok = all_open.ok and chosen.ok
    return {"all_open": all_open, "chosen": chosen,
            "closed": [d for d in network.dcs["dc"] if d not in chosen.open_dcs] if chosen.ok else [],
            "saving": all_open.total_cost - chosen.total_cost if ok else float("nan")}


def minimum_transit_days(network: Network, settings: Settings, longest: int = 10) -> int | None:
    """The tightest service rule (in whole days, up to ``longest``) under which a plan exists, or None."""
    start = 1 if settings.max_transit_days is None else int(settings.max_transit_days)
    for days in range(start, longest + 1):
        if optimize(network, replace(settings, max_transit_days=days)).ok:
            return days
    return None


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", or ".join(items)


def diagnose(network: Network, settings: Settings) -> list[str]:
    """When no plan exists, name the rules whose removal alone would make one possible.

    Tries each active rule in turn: the service rule (reporting the
    shortest transit limit that works), DC capacity, single sourcing, and
    the list of allowed modes. Returns an empty list if a plan already
    exists, or if no single rule is to blame (for example the service rule
    and capacity conflict together).
    """
    if optimize(network, settings).ok:
        return []
    fixes = []
    if settings.max_transit_days is not None and optimize(network, replace(settings, max_transit_days=None)).ok:
        days = minimum_transit_days(network, settings)
        fixes.append(f"the service rule (a plan exists at {days} transit days)" if days else "the service rule")
    if settings.max_capacity_use is not None and optimize(network, replace(settings, max_capacity_use=None)).ok:
        fixes.append("DC capacity")
    if settings.single_sourcing and optimize(network, replace(settings, single_sourcing=False)).ok:
        fixes.append("single sourcing")
    if set(settings.modes) != set(MODES) and optimize(network, replace(settings, modes=MODES)).ok:
        fixes.append("the allowed modes")
    return fixes
