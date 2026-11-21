"""Which distribution center serves which customer, by which mode, and what it costs.

``cost_plan`` costs any allocation: the way customers are served today, or a
recommended one. ``optimize`` finds the cheapest allocation that ships all
demand, fits every DC's weekly capacity and delivers within the service rule.
Both use the same rate functions and the same demand, so a plan and the
baseline are always compared like for like.

How a lane (one DC serving one customer) is costed:

* A customer receives ``shipments_per_week`` shipments each week. Each one
  carries that week's pounds from the DC, divided equally.
* Freight is the mode's charge for each of those shipments (see ``rates``),
  added over all weeks. Handling is the DC's cost per unit times the units.
* Transit days come from the mode and the road miles. A lane is within the
  service rule when its transit days do not exceed ``max_transit_days``.
* A customer split between DCs gets a share of its shipments from each, and
  every shipment keeps its usual size. A lane with share 0.4 therefore costs
  0.4 of the freight and handling of serving the whole customer from that
  DC, and adds 0.4 of the customer's units to the DC's load each week. The
  share is not rounded to whole shipments.

Every cost is an estimate from the rates and demand supplied. The difference
between two plans is an estimated saving, not a realized one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .data import Network
from .rates import MODES, RateCard, shipment_cost, shipment_count, transit_days

LANE_COLUMNS = ["customer_id", "city", "state", "region", "dc", "share", "mode", "miles", "transit_days", "within_service",
                "shipments_per_week", "units", "weight_lbs", "shipments", "freight_cost", "handling_cost", "total_cost"]


@dataclass(frozen=True)
class Settings:
    """Rules and assumptions for one plan."""

    max_transit_days: int | None = 4      # service rule: longest acceptable transit, in days; None = no rule
    single_sourcing: bool = False         # True = every customer is served by exactly one DC
    choose_dcs: bool = False              # True = the plan may close DCs to save their fixed cost
    modes: tuple = MODES                  # modes the plan may use
    max_capacity_use: float | None = 1.0  # share of each DC's weekly capacity the plan may use; None = ignore capacity
    rates: RateCard = RateCard()          # freight rates, fuel surcharge and transit speeds
    time_limit: float = 30.0              # seconds the solver may spend on one plan

    def __post_init__(self):
        if self.max_transit_days is not None and (not np.isfinite(self.max_transit_days) or self.max_transit_days < 1):
            raise ValueError("max_transit_days must be at least 1, or None for no service rule")
        modes = tuple(self.modes)
        if not modes or any(m not in MODES for m in modes) or len(set(modes)) != len(modes):
            raise ValueError(f"modes must be one or more of: {', '.join(MODES)}")
        object.__setattr__(self, "modes", tuple(m for m in MODES if m in modes))
        if self.max_capacity_use is not None and (not np.isfinite(self.max_capacity_use) or self.max_capacity_use <= 0):
            raise ValueError("max_capacity_use must be greater than 0, or None to ignore capacity")
        if not isinstance(self.rates, RateCard):
            raise ValueError("rates must be a RateCard")
        if not np.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit must be greater than 0 seconds")

    def with_fuel(self, fuel_surcharge: float) -> "Settings":
        """The same settings with a different fuel surcharge (0.25 = 25%)."""
        return replace(self, rates=replace(self.rates, fuel_surcharge=float(fuel_surcharge)))


@dataclass
class Result:
    """A plan and its estimated cost over the weeks of demand supplied.

    ``total_cost`` = freight + handling + fixed. Fixed cost is the weekly
    fixed cost of every DC kept open, times the number of weeks, whether or
    not the DC ships anything. ``share_within_service`` and
    ``average_transit_days`` are weighted by units.
    """

    status: str                    # optimal, feasible (time limit reached), evaluated, infeasible or error
    message: str
    lanes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LANE_COLUMNS))
    dc_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    freight_cost: float = 0.0
    handling_cost: float = 0.0
    fixed_cost: float = 0.0
    total_cost: float = 0.0
    weeks: int = 0
    units: float = 0.0
    open_dcs: list = field(default_factory=list)
    share_within_service: float = 0.0
    average_transit_days: float = 0.0
    split_customers: int = 0       # customers served by more than one DC
    violations: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the result holds a costed plan."""
        return self.status in ("optimal", "feasible", "evaluated")

    @property
    def cost_breakdown(self) -> dict:
        return {"Freight": self.freight_cost, "Handling": self.handling_cost, "Fixed DC cost": self.fixed_cost}

    @property
    def assignment(self) -> pd.DataFrame:
        """The plan in its smallest form: customer, DC, share and mode."""
        return self.lanes[["customer_id", "dc", "share", "mode"]].reset_index(drop=True)

    @property
    def mode_mix(self) -> pd.DataFrame:
        """Shipments, pounds and freight cost by mode."""
        if self.lanes.empty:
            return pd.DataFrame(columns=["mode", "lanes", "shipments", "weight_lbs", "freight_cost"])
        mix = self.lanes.groupby("mode").agg(lanes=("dc", "size"), shipments=("shipments", "sum"), weight_lbs=("weight_lbs", "sum"),
                                             freight_cost=("freight_cost", "sum"))
        return mix.reindex([m for m in MODES if m in mix.index]).reset_index()


class _Model:
    """The network as arrays, with the lane costing used by every function in the package."""

    def __init__(self, network: Network, settings: Settings):
        if not isinstance(network, Network):
            raise ValueError("network must be a Network; build one with validate_network or load_network")
        if not isinstance(settings, Settings):
            raise ValueError("settings must be a Settings instance")
        self.network, self.settings, self.rates = network, settings, settings.rates
        self.dcs, self.customers = network.dcs, network.customers
        self.dc_names = network.dcs["dc"].tolist()
        self.ids = network.customers["customer_id"].tolist()
        self.miles = network.distances()                                       # DC x customer
        self.units, self.weight = network.weekly()                             # customer x week
        self.per_week = network.customers["shipments_per_week"].to_numpy(dtype=float)
        self.weeks = len(network.weeks)
        self.handling = network.dcs["handling_cost_per_unit"].to_numpy(dtype=float)
        self.fixed = network.dcs["fixed_cost_per_week"].to_numpy(dtype=float)
        use = settings.max_capacity_use
        self.capacity = None if use is None else network.dcs["capacity_units_per_week"].to_numpy(dtype=float) * use

    def freight(self, mode: str, i, c, per_week=None) -> np.ndarray:
        """Freight over all weeks if DC ``i`` serves all of customer ``c`` (arrays of lanes)."""
        n = (self.per_week[c] if per_week is None else np.asarray(per_week, dtype=float))[:, None]
        return (n * shipment_cost(mode, self.miles[i, c][:, None], self.weight[c] / n, self.rates)).sum(axis=1)

    def shipments(self, mode: str, c, per_week=None) -> np.ndarray:
        """Packages, LTL shipments or trucks over all weeks if one DC serves all of customer ``c``."""
        n = (self.per_week[c] if per_week is None else np.asarray(per_week, dtype=float))[:, None]
        return (n * shipment_count(mode, self.weight[c] / n, self.rates)).sum(axis=1)

    def transit(self, mode: str, i, c) -> np.ndarray:
        return transit_days(mode, self.miles[i, c], self.rates)

    def best_modes(self, i, c, per_week=None, extra_days=0):
        """Cheapest allowed mode per lane that meets the service rule.

        Returns ``(mode index into MODES, freight for the whole customer, transit days, meets rule)``.
        Where no allowed mode meets the rule, the cheapest allowed mode is
        returned with ``meets rule`` False. ``extra_days`` is waiting time
        added to transit before the rule is checked.
        """
        i, c = np.asarray(i), np.asarray(c)
        limit = self.settings.max_transit_days
        best = np.full(len(i), np.inf)
        fallback = np.full(len(i), np.inf)
        pick, pick_any = np.zeros(len(i), dtype=int), np.zeros(len(i), dtype=int)
        for mode in self.settings.modes:
            k = MODES.index(mode)
            cost, days = self.freight(mode, i, c, per_week), self.transit(mode, i, c)
            ok = np.ones(len(i), dtype=bool) if limit is None else days + extra_days <= limit
            better = ok & (cost < best)
            best[better], pick[better] = cost[better], k
            cheaper = cost < fallback
            fallback[cheaper], pick_any[cheaper] = cost[cheaper], k
        meets = np.isfinite(best)
        pick = np.where(meets, pick, pick_any)
        cost = np.where(meets, best, fallback)
        days = np.zeros(len(i), dtype=int)
        for k, mode in enumerate(MODES):
            hit = pick == k
            if hit.any():
                days[hit] = self.transit(mode, i[hit], c[hit])
        return pick, cost, days, meets


def lane_options(network: Network, settings: Settings = Settings()) -> pd.DataFrame:
    """Every DC, customer and mode: what the lane would cost if that DC served all of the customer's demand.

    One row per combination with ``miles``, ``transit_days``, whether it
    meets the service rule, ``freight_cost`` and ``handling_cost`` over all
    weeks, and ``cheapest`` marking the cheapest mode on the lane that meets
    the rule. Modes not allowed by the settings are left out.
    """
    m = _Model(network, settings)
    i, c = (a.ravel() for a in np.meshgrid(np.arange(len(m.dc_names)), np.arange(len(m.ids)), indexing="ij"))
    pick, _, _, meets = m.best_modes(i, c)
    limit = settings.max_transit_days
    frames = []
    for mode in settings.modes:
        days = m.transit(mode, i, c)
        frames.append(pd.DataFrame({
            "dc": np.array(m.dc_names)[i], "customer_id": np.array(m.ids)[c], "mode": mode, "miles": np.round(m.miles[i, c], 0),
            "transit_days": days, "within_service": np.ones(len(i), dtype=bool) if limit is None else days <= limit,
            "freight_cost": m.freight(mode, i, c), "handling_cost": m.handling[i] * m.units[c].sum(axis=1),
            "cheapest": meets & (pick == MODES.index(mode))}))
    out = pd.concat(frames, ignore_index=True)
    out["total_cost"] = out["freight_cost"] + out["handling_cost"]
    return out.sort_values(["customer_id", "dc", "mode"]).reset_index(drop=True)


def cost_plan(network: Network, assignment: pd.DataFrame, settings: Settings = Settings(), open_dcs=None,
              status: str = "evaluated", message: str = "Plan costed as given.") -> Result:
    """Cost an allocation exactly as given. Nothing is re-optimized.

    ``assignment`` needs ``customer_id`` and ``dc``; ``share`` (default 1)
    and ``mode`` are optional. Where the mode is blank, the cheapest allowed
    mode that meets the service rule is used (the cheapest allowed mode if
    none meets it). ``open_dcs`` lists the DCs that pay their fixed cost;
    by default every DC in the network.

    The result lists in ``violations`` every rule the allocation breaks under
    these settings: customers not fully served, DCs that do not exist or are
    closed, weekly capacity exceeded, and lanes slower than the service rule.
    """
    m = _Model(network, settings)
    missing = [col for col in ("customer_id", "dc") if col not in assignment.columns]
    if missing:
        raise ValueError(f"assignment is missing required column(s): {', '.join(missing)}")
    plan = assignment.copy()
    plan["customer_id"], plan["dc"] = plan["customer_id"].astype(str).str.strip(), plan["dc"].astype(str).str.strip()
    plan["share"] = pd.to_numeric(plan["share"], errors="coerce").fillna(1.0) if "share" in plan.columns else 1.0
    plan["mode"] = plan["mode"].fillna("").astype(str).str.strip() if "mode" in plan.columns else ""
    bad_mode = sorted(set(plan["mode"]) - set(MODES) - {""})
    if bad_mode:
        raise ValueError(f"assignment has unknown mode(s): {', '.join(bad_mode)}")
    if (plan["share"] < 0).any():
        raise ValueError("assignment shares cannot be negative")
    plan = plan[plan["share"] > 0]
    open_dcs = list(m.dc_names) if open_dcs is None else [d for d in m.dc_names if d in set(open_dcs)]

    violations = []
    unknown_customer = ~plan["customer_id"].isin(m.ids)
    if unknown_customer.any():
        violations.append(f"{int(unknown_customer.sum())} row(s) refer to a customer that is not in the network")
    unknown_dc = ~plan["dc"].isin(m.dc_names)
    if unknown_dc.any():
        gone = sorted(plan.loc[unknown_dc, "dc"].unique())
        violations.append(f"{plan.loc[unknown_dc, 'customer_id'].nunique()} customer(s) are assigned to a DC that is not in the network: {', '.join(gone)}")
    plan = plan[~unknown_customer & ~unknown_dc]
    plan = plan.groupby(["customer_id", "dc"], as_index=False, sort=False).agg(share=("share", "sum"), mode=("mode", "first"))

    i = plan["dc"].map({d: k for k, d in enumerate(m.dc_names)}).to_numpy(dtype=int)
    c = plan["customer_id"].map({d: k for k, d in enumerate(m.ids)}).to_numpy(dtype=int)
    share = plan["share"].to_numpy(dtype=float)
    pick, freight, days, _ = m.best_modes(i, c)
    given = plan["mode"].to_numpy(dtype=object)
    for mode in MODES:
        hit = given == mode
        if hit.any():
            pick[hit] = MODES.index(mode)
            freight[hit] = m.freight(mode, i[hit], c[hit])
            days[hit] = m.transit(mode, i[hit], c[hit])
    modes = np.array(MODES, dtype=object)[pick] if len(pick) else np.array([], dtype=object)
    shipments = np.zeros(len(i))
    for mode in MODES:
        hit = modes == mode
        if hit.any():
            shipments[hit] = m.shipments(mode, c[hit])
    freight, shipments = share * freight, share * shipments
    limit = settings.max_transit_days
    units = share * m.units[c].sum(axis=1)
    handling = m.handling[i] * units
    cust = m.customers.iloc[c]
    lanes = pd.DataFrame({
        "customer_id": plan["customer_id"].to_numpy(), "city": cust["city"].to_numpy(), "state": cust["state"].to_numpy(),
        "region": cust["region"].to_numpy(), "dc": plan["dc"].to_numpy(), "share": share, "mode": modes,
        "miles": np.round(m.miles[i, c], 0), "transit_days": days,
        "within_service": np.ones(len(i), dtype=bool) if limit is None else days <= limit,
        "shipments_per_week": m.per_week[c].astype(int), "units": units, "weight_lbs": share * m.weight[c].sum(axis=1),
        "shipments": shipments, "freight_cost": freight, "handling_cost": handling, "total_cost": freight + handling,
    }, columns=LANE_COLUMNS).sort_values(["customer_id", "share", "dc"], ascending=[True, False, True]).reset_index(drop=True)

    # Rules.
    served = pd.Series(share, index=plan["customer_id"].to_numpy()).groupby(level=0).sum().reindex(m.ids).fillna(0.0)
    short = int(((served - 1.0).abs() > 1e-6).sum())
    if short:
        violations.append(f"{short} customer(s) are not fully served (shares do not add to 1)")
    closed = sorted(set(plan["dc"]) - set(open_dcs))
    if closed:
        violations.append(f"customers are assigned to closed DC(s): {', '.join(closed)}")
    weekly = np.zeros((len(m.dc_names), m.weeks))
    np.add.at(weekly, i, share[:, None] * m.units[c])
    peak = weekly.max(axis=1) if m.weeks else np.zeros(len(m.dc_names))
    capacity = m.dcs["capacity_units_per_week"].to_numpy(dtype=float)
    if m.capacity is not None:
        for k in np.flatnonzero(peak > m.capacity * (1 + 1e-9) + 1e-6):
            violations.append(f"{m.dc_names[k]}: busiest week ships {peak[k]:,.0f} units, above the {m.capacity[k]:,.0f} allowed")
    late = lanes[~lanes["within_service"]]
    if len(late):
        violations.append(f"{late['customer_id'].nunique()} customer(s) are served slower than the {limit}-day service rule")

    is_open = np.array([d in set(open_dcs) for d in m.dc_names])
    by_dc = lanes.groupby("dc").agg(customers=("customer_id", "nunique"), units=("units", "sum"), freight_cost=("freight_cost", "sum"),
                                    handling_cost=("handling_cost", "sum")).reindex(m.dc_names).fillna(0.0)
    dc_summary = pd.DataFrame({
        "dc": m.dc_names, "open": is_open, "customers": by_dc["customers"].to_numpy().astype(int), "units": by_dc["units"].to_numpy(),
        "peak_week_units": peak, "capacity_units_per_week": capacity,
        "peak_utilization": np.divide(peak, capacity, out=np.zeros(len(peak)), where=capacity > 0),
        "freight_cost": by_dc["freight_cost"].to_numpy(), "handling_cost": by_dc["handling_cost"].to_numpy(),
        "fixed_cost": np.where(is_open, m.fixed * m.weeks, 0.0)})
    total_units = float(lanes["units"].sum())
    freight_total, handling_total, fixed_total = float(freight.sum()), float(handling.sum()), float(dc_summary["fixed_cost"].sum())
    return Result(
        status=status, message=message, lanes=lanes, dc_summary=dc_summary,
        freight_cost=freight_total, handling_cost=handling_total, fixed_cost=fixed_total,
        total_cost=freight_total + handling_total + fixed_total, weeks=m.weeks, units=total_units, open_dcs=open_dcs,
        share_within_service=float(lanes.loc[lanes["within_service"], "units"].sum()) / total_units if total_units else 0.0,
        average_transit_days=float((lanes["units"] * lanes["transit_days"]).sum()) / total_units if total_units else 0.0,
        split_customers=int((lanes.groupby("customer_id")["dc"].size() > 1).sum()) if len(lanes) else 0,
        violations=violations)


def baseline_plan(network: Network, settings: Settings = Settings()) -> Result:
    """Today's allocation (the ``baseline`` table), costed with the same rates and demand as any other plan.

    Every DC in the network is counted as open. Rules that today's
    allocation breaks under these settings are listed in ``violations``.
    """
    return cost_plan(network, network.baseline, settings, status="evaluated", message="Today's allocation, costed as given.")


def baseline_with_best_modes(network: Network, settings: Settings = Settings()) -> Result:
    """Today's DC for every customer, with the cheapest allowed mode on each lane whatever its transit time.

    Shows how much of a saving is available from mode choice alone, with no
    customer moving to another DC. The service rule is not applied when the
    mode is picked (today's lanes may be too long to meet it by any mode),
    but the result still reports service against the rule.
    """
    free = replace(settings, max_transit_days=None)
    modes = cost_plan(network, network.baseline[["customer_id", "dc", "share"]], free).assignment
    return cost_plan(network, modes, settings, message="Today's DCs with the cheapest mode on each lane.")


def _solve(m: _Model, use_open: bool, n_open: int | None):
    """Build and solve the allocation model. Returns ``(status, message, plan, open DCs)``."""
    settings = m.settings
    single, time_limit = settings.single_sourcing, float(settings.time_limit)
    n_dc, n_cust = len(m.dc_names), len(m.ids)
    i_all, c_all = (a.ravel() for a in np.meshgrid(np.arange(n_dc), np.arange(n_cust), indexing="ij"))
    _, freight, _, meets = m.best_modes(i_all, c_all)
    total_units = m.units.sum(axis=1)
    li, lc = i_all[meets], c_all[meets]
    lane_cost = freight[meets] + m.handling[li] * total_units[lc]

    unreachable = np.setdiff1d(np.arange(n_cust), lc)
    if len(unreachable):
        names = [f"{m.ids[k]} ({m.customers['city'].iloc[k] or m.customers['state'].iloc[k]})" for k in unreachable[:5]]
        more = f" and {len(unreachable) - 5} more" if len(unreachable) > 5 else ""
        return ("infeasible", f"{len(unreachable)} customer(s) cannot be reached within {settings.max_transit_days} day(s) from any DC "
                              f"with the allowed modes: {', '.join(names)}{more}. Relax the service rule or allow a faster mode.", None, [])
    if m.capacity is not None:
        need, have = m.units.sum(axis=0).max(), m.capacity.sum()
        if need > have * (1 + 1e-9):
            return ("infeasible", f"Demand in the busiest week ({need:,.0f} units) is more than the capacity of all DCs together "
                                  f"({have:,.0f} units). Add capacity or reduce demand.", None, [])

    n_lane = len(li)
    lanes, opened = np.arange(n_lane), n_lane + np.arange(n_dc)
    n_var = n_lane + (n_dc if use_open else 0)
    # Constraint rows are collected as (row, column, value) triplets with a lower and upper limit per row.
    rows, cols, vals = [lc], [lanes], [np.ones(n_lane)]              # every customer fully served: shares add to 1
    low, high = [1.0] * n_cust, [1.0] * n_cust
    r = n_cust
    if m.capacity is not None:
        # In every week, the units a DC ships fit its capacity (zero when the DC is closed).
        for w in range(m.weeks):
            rows += [r + li]
            cols += [lanes]
            vals += [m.units[lc, w]]
            if use_open:
                rows += [r + np.arange(n_dc)]
                cols += [opened]
                vals += [-m.capacity]
            low += [-np.inf] * n_dc
            high += [0.0] * n_dc if use_open else list(m.capacity)
            r += n_dc
    if use_open:
        # A lane can be used only if its DC is open.
        rows += [r + lanes, r + lanes]
        cols += [lanes, n_lane + li]
        vals += [np.ones(n_lane), -np.ones(n_lane)]
        low += [-np.inf] * n_lane
        high += [0.0] * n_lane
        r += n_lane
        if n_open is not None:
            rows += [np.full(n_dc, r)]
            cols += [opened]
            vals += [np.ones(n_dc)]
            low += [float(n_open)]
            high += [float(n_open)]
            r += 1
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n_var)).tocsr()
    cost = np.r_[lane_cost, m.fixed * m.weeks] if use_open else lane_cost
    integrality = np.r_[np.full(n_lane, 1 if single else 0), np.ones(n_var - n_lane)]
    solved = milp(cost, integrality=integrality, bounds=Bounds(np.zeros(n_var), np.ones(n_var)),
                  constraints=LinearConstraint(A, np.array(low), np.array(high)), options={"time_limit": time_limit})
    if solved.x is None:
        if solved.status == 2:
            return "infeasible", "No allocation satisfies all of the rules together.", None, []
        if solved.status == 1:
            return "error", (f"No allocation was found within the time limit of {time_limit:g} seconds. One DC per customer is hard "
                             "to fit when capacity is nearly full; allow more time or allow customers to be split."), None, []
        return "error", f"The solver did not finish: {solved.message}", None, []

    share = np.clip(solved.x[:n_lane], 0.0, 1.0)
    share[share < 1e-7] = 0.0
    if single:
        share = np.round(share)
    used = share > 0
    plan = pd.DataFrame({"customer_id": np.array(m.ids)[lc[used]], "dc": np.array(m.dc_names)[li[used]], "share": share[used]})
    plan["share"] = plan["share"] / plan.groupby("customer_id")["share"].transform("sum")
    open_dcs = [d for k, d in enumerate(m.dc_names) if solved.x[n_lane + k] > 0.5] if use_open else list(m.dc_names)
    if solved.status == 0:
        return "optimal", "Optimal allocation found.", plan, open_dcs
    return "feasible", "Time limit reached; this is the best allocation found so far.", plan, open_dcs


def optimize(network: Network, settings: Settings = Settings(), n_open: int | None = None) -> Result:
    """Find the cheapest allocation of customers to DCs that obeys every rule.

    Minimizes freight + handling over the weeks of demand, plus the fixed
    cost of the DCs kept open when ``settings.choose_dcs`` is set. Rules:

    * every customer's demand is served in full;
    * in every week, the units a DC ships stay within its capacity;
    * a lane is used only if some allowed mode delivers within
      ``max_transit_days``; each lane uses the cheapest such mode;
    * with ``single_sourcing``, each customer has exactly one DC;
    * with ``choose_dcs``, a closed DC ships nothing and pays no fixed cost.
      Otherwise every DC stays open and pays its fixed cost.

    ``n_open`` forces an exact number of open DCs (and implies choosing which).

    Without ``single_sourcing`` and without DC choices this is a linear
    program and solves at once. With them it is a mixed-integer program,
    solved with HiGHS to its default tolerance (0.01% of cost); when DC
    capacity is nearly full, one DC per customer can take long to prove
    optimal, and ``settings.time_limit`` then returns the best plan found.

    Returns a ``Result`` with status ``optimal``, ``feasible`` (time limit
    reached with a plan in hand), ``infeasible`` or ``error``.
    """
    m = _Model(network, settings)
    use_open = settings.choose_dcs or n_open is not None
    if n_open is not None and not (int(n_open) == n_open and 1 <= int(n_open) <= len(m.dc_names)):
        raise ValueError(f"n_open must be a whole number between 1 and {len(m.dc_names)}")

    status, message, plan, open_dcs = _solve(m, use_open, n_open)
    if plan is None:
        return Result(status, message)
    return cost_plan(network, plan, settings, open_dcs=open_dcs, status=status, message=message)


# ---------------------------------------------------------------- comparison
@dataclass
class Comparison:
    """Today's allocation against a recommended one, costed like for like.

    ``saving`` is baseline total cost minus optimized total cost over
    ``weeks`` weeks of demand: an estimate under the rates and demand
    supplied, not a realized saving.
    """

    costs: pd.DataFrame            # component, baseline, optimized, change, change_pct
    saving: float
    saving_pct: float
    weeks: int
    lane_changes: pd.DataFrame     # one row per customer whose DC or mode changes
    service: pd.DataFrame          # measure, baseline, optimized
    customers: int
    customers_changed: int
    dc_changes: int                # customers that move to a different DC
    mode_changes: int              # customers whose mode changes

    @property
    def saving_per_week(self) -> float:
        return self.saving / self.weeks if self.weeks else 0.0


def _by_customer(result: Result) -> pd.DataFrame:
    """One row per customer: DC and mode as text, cost, and the slowest lane."""
    lanes = result.lanes

    def label(group: pd.DataFrame, column: str) -> str:
        if len(group) == 1:
            return str(group[column].iloc[0])
        return " + ".join(f"{v} {s:.0%}" for v, s in zip(group[column], group["share"]))

    rows = []
    for customer, g in lanes.groupby("customer_id", sort=True):
        rows.append({"customer_id": customer, "city": g["city"].iloc[0], "state": g["state"].iloc[0], "region": g["region"].iloc[0],
                     "dc": label(g, "dc"), "mode": label(g, "mode") if g["mode"].nunique() > 1 else str(g["mode"].iloc[0]),
                     "miles": float((g["miles"] * g["share"]).sum()), "transit_days": int(g["transit_days"].max()),
                     "units": float(g["units"].sum()), "weight_lbs": float(g["weight_lbs"].sum()),
                     "freight_cost": float(g["freight_cost"].sum()), "handling_cost": float(g["handling_cost"].sum()),
                     "cost": float(g["total_cost"].sum())})
    return pd.DataFrame(rows, columns=["customer_id", "city", "state", "region", "dc", "mode", "miles", "transit_days", "units",
                                       "weight_lbs", "freight_cost", "handling_cost", "cost"])


def compare(baseline: Result, optimized: Result) -> Comparison:
    """Compare two costed plans of the same network: costs, the lanes that change, and service.

    ``lane_changes`` has one row per customer whose DC or mode differs, with
    ``cost_before``, ``cost_after`` (freight + handling for that customer)
    and ``saving``, largest saving first. A single customer can cost more in
    the recommended plan when it gives up DC capacity to others or moves to
    meet the service rule; what is minimized is the total.
    """
    if not (baseline.ok and optimized.ok):
        raise ValueError("both plans must be costed before they can be compared; check their status and message")
    if baseline.weeks != optimized.weeks or abs(baseline.units - optimized.units) > 1e-6 * max(baseline.units, 1.0):
        raise ValueError("the two plans do not cover the same demand, so they cannot be compared like for like")
    rows = [(name, before, optimized.cost_breakdown[name]) for name, before in baseline.cost_breakdown.items()]
    rows.append(("Total", baseline.total_cost, optimized.total_cost))
    costs = pd.DataFrame(rows, columns=["component", "baseline", "optimized"])
    costs["change"] = costs["optimized"] - costs["baseline"]
    costs["change_pct"] = np.divide(costs["change"], costs["baseline"], out=np.zeros(len(costs)), where=costs["baseline"].to_numpy() != 0)

    a, b = _by_customer(baseline), _by_customer(optimized)
    both = a.merge(b, on=["customer_id", "city", "state", "region"], suffixes=("_before", "_after"))
    dc_moved, mode_moved = both["dc_before"] != both["dc_after"], both["mode_before"] != both["mode_after"]
    both["change"] = np.select([dc_moved & mode_moved, dc_moved, mode_moved], ["DC and mode", "DC", "mode"], "none")
    both["saving"] = both["cost_before"] - both["cost_after"]
    changes = both[both["change"] != "none"].rename(columns={
        "dc_before": "from_dc", "dc_after": "to_dc", "mode_before": "from_mode", "mode_after": "to_mode",
        "units_before": "units", "weight_lbs_before": "weight_lbs"})
    changes = changes[["customer_id", "city", "state", "region", "change", "from_dc", "to_dc", "from_mode", "to_mode", "units", "weight_lbs",
                       "miles_before", "miles_after", "transit_days_before", "transit_days_after", "cost_before", "cost_after", "saving"]]
    changes = changes.sort_values(["saving", "customer_id"], ascending=[False, True]).reset_index(drop=True)

    service = pd.DataFrame([
        ("Share of units delivered within the service rule", baseline.share_within_service, optimized.share_within_service),
        ("Average transit days (weighted by units)", baseline.average_transit_days, optimized.average_transit_days),
        ("Average road miles per unit", float((baseline.lanes["miles"] * baseline.lanes["units"]).sum()) / max(baseline.units, 1e-9),
         float((optimized.lanes["miles"] * optimized.lanes["units"]).sum()) / max(optimized.units, 1e-9)),
        ("DCs open", float(len(baseline.open_dcs)), float(len(optimized.open_dcs))),
    ], columns=["measure", "baseline", "optimized"])
    saving = baseline.total_cost - optimized.total_cost
    return Comparison(costs=costs, saving=saving, saving_pct=saving / baseline.total_cost if baseline.total_cost else 0.0,
                      weeks=baseline.weeks, lane_changes=changes, service=service, customers=len(a),
                      customers_changed=int((both["change"] != "none").sum()), dc_changes=int(dc_moved.sum()),
                      mode_changes=int(mode_moved.sum()))
