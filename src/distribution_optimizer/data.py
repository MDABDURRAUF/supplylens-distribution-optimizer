"""The distribution network: expected tables, validation, files, and a simulated US sample.

A network is five tables:

``dcs``        one row per distribution center
``customers``  one row per customer location
``products``   one row per product family, with its weight per unit
``demand``     units by week, customer and product family
``baseline``   how each customer is served today (which DC, which mode)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .rates import MODES
from .us import ALL_METROS, STATE_REGION, road_miles

TABLES = ("dcs", "customers", "products", "demand", "baseline")
REQUIRED = {
    "dcs": ["dc", "state", "latitude", "longitude", "capacity_units_per_week", "handling_cost_per_unit"],
    "customers": ["customer_id", "state", "latitude", "longitude"],
    "products": ["product_family", "weight_lbs_per_unit"],
    "demand": ["week", "customer_id", "product_family", "units"],
    "baseline": ["customer_id", "dc"],
}
# Optional columns and the value used when a column is absent or a cell is blank.
OPTIONAL = {
    "dcs": {"fixed_cost_per_week": 0.0},
    "customers": {"city": "", "shipments_per_week": 1},
    "products": {},
    "demand": {},
    "baseline": {"mode": "", "share": 1.0},
}
# What every column means, for the README, the dashboard and error messages.
COLUMN_HELP = [
    ("dcs", "dc", "yes", "Name of the distribution center (unique)"),
    ("dcs", "state", "yes", "Two-letter US state code"),
    ("dcs", "latitude, longitude", "yes", "Location in degrees"),
    ("dcs", "capacity_units_per_week", "yes", "Most units the DC can ship in one week, all product families together"),
    ("dcs", "handling_cost_per_unit", "yes", "Cost of picking, packing and loading one unit, in dollars"),
    ("dcs", "fixed_cost_per_week", "no", "Cost of keeping the DC open for a week, whatever it ships (default 0)"),
    ("customers", "customer_id", "yes", "Unique identifier"),
    ("customers", "state", "yes", "Two-letter US state code"),
    ("customers", "latitude, longitude", "yes", "Location in degrees"),
    ("customers", "city", "no", "City or site name, for display"),
    ("customers", "shipments_per_week", "no", "How many separate shipments the customer receives in a week today (default 1)"),
    ("products", "product_family", "yes", "Name of the product family (unique)"),
    ("products", "weight_lbs_per_unit", "yes", "Shipping weight of one unit, in pounds"),
    ("demand", "week", "yes", "Week label, for example the Monday date"),
    ("demand", "customer_id, product_family", "yes", "Must exist in the customers and products tables"),
    ("demand", "units", "yes", "Units ordered that week"),
    ("baseline", "customer_id, dc", "yes", "Which DC serves the customer today"),
    ("baseline", "mode", "no", "parcel, LTL or truckload; blank means the cheapest mode is assumed"),
    ("baseline", "share", "no", "Share of the customer's demand from this DC (default 1; rows of a customer must add to 1)"),
]


class SchemaError(ValueError):
    """Raised when a network table does not match the expected layout."""


@dataclass
class Network:
    """A validated network. Build one with ``validate_network``, ``load_network`` or ``generate_sample_network``."""

    dcs: pd.DataFrame
    customers: pd.DataFrame
    products: pd.DataFrame
    demand: pd.DataFrame
    baseline: pd.DataFrame

    @property
    def weeks(self) -> list:
        return sorted(self.demand["week"].unique())

    def weekly(self) -> tuple[np.ndarray, np.ndarray]:
        """Units and pounds by customer (rows, in ``customers`` order) and week (columns, in ``weeks`` order)."""
        d = self.demand.merge(self.products, on="product_family")
        d["weight"] = d["units"] * d["weight_lbs_per_unit"]
        grid = d.pivot_table(index="customer_id", columns="week", values=["units", "weight"], aggfunc="sum", fill_value=0.0)
        ids = self.customers["customer_id"]
        units = grid["units"].reindex(index=ids, columns=self.weeks).fillna(0.0).to_numpy(dtype=float)
        weight = grid["weight"].reindex(index=ids, columns=self.weeks).fillna(0.0).to_numpy(dtype=float)
        return units, weight

    def distances(self) -> np.ndarray:
        """Estimated road miles from every DC (rows) to every customer (columns), at least ``LOCAL_MILES``."""
        d, c = self.dcs, self.customers
        miles = road_miles(d["latitude"].to_numpy()[:, None], d["longitude"].to_numpy()[:, None],
                           c["latitude"].to_numpy()[None, :], c["longitude"].to_numpy()[None, :])
        return np.maximum(miles, LOCAL_MILES)

    def with_demand_growth(self, growth: float) -> "Network":
        """The same network with every demand line changed by ``growth`` (0.10 = 10% more), in whole units."""
        if not np.isfinite(growth) or growth <= -1:
            raise ValueError("growth must be greater than -1 (which would remove all demand)")
        demand = self.demand.copy()
        demand["units"] = np.round(demand["units"] * (1.0 + growth))
        return replace(self, demand=demand)

    def without_dc(self, dc: str) -> "Network":
        """The same network with one DC removed. The baseline is kept, so it will refer to a missing DC."""
        if dc not in set(self.dcs["dc"]):
            raise ValueError(f"'{dc}' is not a DC in this network")
        if len(self.dcs) == 1:
            raise ValueError("the network has only one DC")
        return replace(self, dcs=self.dcs[self.dcs["dc"] != dc].reset_index(drop=True))

    def save(self, folder: str | Path) -> Path:
        """Write the five tables as CSV files into ``folder`` (created if needed)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            getattr(self, name).drop(columns=["region"], errors="ignore").to_csv(folder / f"{name}.csv", index=False)
        return folder


# A customer in the same city as a DC still needs a local delivery; distances are never shorter than this.
LOCAL_MILES = 15.0


def _text(frame: pd.DataFrame, column: str, table: str, problems: list, upper: bool = False) -> None:
    blank = frame[column].isna()
    frame[column] = frame[column].astype(str).str.strip()
    if upper:
        frame[column] = frame[column].str.upper()
    blank |= frame[column].eq("")
    if blank.any():
        problems.append(f"{table}: {int(blank.sum())} row(s) with a blank '{column}'")


def _numbers(frame: pd.DataFrame, columns, table: str, problems: list, positive=()) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            problems.append(f"{table}: {int(bad.sum())} row(s) with a missing or unreadable '{column}'")
        elif (values < 0).any():
            problems.append(f"{table}: {int((values < 0).sum())} row(s) with a negative '{column}'")
        elif column in positive and (values <= 0).any():
            problems.append(f"{table}: {int((values <= 0).sum())} row(s) where '{column}' is not greater than zero")


def _places(frame: pd.DataFrame, table: str, problems: list) -> None:
    bad_state = ~frame["state"].isin(STATE_REGION)
    if bad_state.any():
        problems.append(f"{table}: {int(bad_state.sum())} row(s) with a state code that is not a US state")
    for column, limit in (("latitude", 90), ("longitude", 180)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        bad = ~(frame[column].abs() <= limit)
        if bad.any():
            problems.append(f"{table}: {int(bad.sum())} row(s) with a missing or impossible '{column}'")
    frame["region"] = frame["state"].map(STATE_REGION)


def validate_network(dcs: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame, demand: pd.DataFrame,
                     baseline: pd.DataFrame) -> Network:
    """Check the five tables against each other and return a cleaned ``Network``.

    Text is trimmed, numbers are parsed, optional columns are added with
    their defaults, and a Census ``region`` is added to DCs and customers.
    Demand lines for the same week, customer and family are added together.
    Every problem found is listed in one ``SchemaError`` with the number of
    rows affected; nothing is dropped silently.
    """
    frames = {"dcs": dcs, "customers": customers, "products": products, "demand": demand, "baseline": baseline}
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError(f"{name} must be a table (pandas DataFrame)")
        missing = [c for c in REQUIRED[name] if c not in frame.columns]
        if missing:
            raise SchemaError(f"{name} is missing required column(s): {', '.join(missing)}")
        if frame.empty:
            raise SchemaError(f"{name} has no rows")
    out = {name: frame.copy().reset_index(drop=True) for name, frame in frames.items()}
    for name, defaults in OPTIONAL.items():
        for column, default in defaults.items():
            if column not in out[name].columns:
                out[name][column] = default
    d, c, p, q, b = (out[name] for name in TABLES)
    problems: list[str] = []

    _text(d, "dc", "dcs", problems)
    _text(d, "state", "dcs", problems, upper=True)
    _text(c, "customer_id", "customers", problems)
    _text(c, "state", "customers", problems, upper=True)
    _text(p, "product_family", "products", problems)
    for column in ("week", "customer_id", "product_family"):
        _text(q, column, "demand", problems)
    _text(b, "customer_id", "baseline", problems)
    _text(b, "dc", "baseline", problems)
    c["city"] = c["city"].fillna("").astype(str).str.strip()
    b["mode"] = b["mode"].fillna("").astype(str).str.strip()
    lookup = {m.lower(): m for m in MODES}
    b["mode"] = b["mode"].map(lambda m: lookup.get(m.lower(), m))

    d["fixed_cost_per_week"] = pd.to_numeric(d["fixed_cost_per_week"], errors="coerce").fillna(0.0)
    c["shipments_per_week"] = pd.to_numeric(c["shipments_per_week"], errors="coerce").fillna(1)
    b["share"] = pd.to_numeric(b["share"], errors="coerce").fillna(1.0)
    _numbers(d, ["capacity_units_per_week", "handling_cost_per_unit", "fixed_cost_per_week"], "dcs", problems)
    _numbers(c, ["shipments_per_week"], "customers", problems, positive=["shipments_per_week"])
    _numbers(p, ["weight_lbs_per_unit"], "products", problems, positive=["weight_lbs_per_unit"])
    _numbers(q, ["units"], "demand", problems)
    _numbers(b, ["share"], "baseline", problems, positive=["share"])
    _places(d, "dcs", problems)
    _places(c, "customers", problems)

    for frame, column, name in ((d, "dc", "dcs"), (c, "customer_id", "customers"), (p, "product_family", "products")):
        if frame[column].duplicated().any():
            problems.append(f"{name}: {int(frame[column].duplicated().sum())} duplicate '{column}' value(s)")
    whole = np.isfinite(c["shipments_per_week"]) & (c["shipments_per_week"] != np.floor(c["shipments_per_week"]))
    if whole.any():
        problems.append(f"customers: {int(whole.sum())} row(s) where 'shipments_per_week' is not a whole number")
    for column, parent, name in (("customer_id", c, "customers"), ("product_family", p, "products")):
        unknown = ~q[column].isin(parent[column])
        if unknown.any():
            problems.append(f"demand: {int(unknown.sum())} row(s) whose '{column}' is not in the {name} table")
    unknown = ~b["customer_id"].isin(c["customer_id"])
    if unknown.any():
        problems.append(f"baseline: {int(unknown.sum())} row(s) whose 'customer_id' is not in the customers table")
    unknown = ~b["dc"].isin(d["dc"])
    if unknown.any():
        problems.append(f"baseline: {int(unknown.sum())} row(s) whose 'dc' is not in the dcs table")
    bad_mode = ~b["mode"].isin(("",) + MODES)
    if bad_mode.any():
        problems.append(f"baseline: {int(bad_mode.sum())} row(s) with a mode other than {', '.join(MODES)}")
    if b.duplicated(["customer_id", "dc"]).any():
        problems.append(f"baseline: {int(b.duplicated(['customer_id', 'dc']).sum())} repeated customer and DC pair(s)")
    if problems:
        raise SchemaError("Invalid network data: " + "; ".join(problems))

    # Week labels that are all dates are written the same way, so they sort in time order.
    as_date = pd.to_datetime(q["week"], errors="coerce", format="%Y-%m-%d")
    if as_date.notna().all():
        q["week"] = as_date.dt.strftime("%Y-%m-%d")
    q = q.groupby(["week", "customer_id", "product_family"], as_index=False, sort=True)["units"].sum()
    ordered = q.groupby("customer_id")["units"].sum().reindex(c["customer_id"]).fillna(0.0)
    if (ordered <= 0).any():
        problems.append(f"customers: {int((ordered <= 0).sum())} customer(s) with no demand; remove them or add demand")
    shares = b.groupby("customer_id")["share"].sum().reindex(c["customer_id"])
    if shares.isna().any():
        problems.append(f"baseline: {int(shares.isna().sum())} customer(s) with no row saying how they are served today")
    if ((shares - 1.0).abs() > 1e-6).any():
        problems.append(f"baseline: {int(((shares - 1.0).abs() > 1e-6).sum())} customer(s) whose shares do not add to 1")
    if problems:
        raise SchemaError("Invalid network data: " + "; ".join(problems))
    c["shipments_per_week"] = c["shipments_per_week"].astype(int)
    return Network(dcs=d, customers=c, products=p, demand=q, baseline=b)


def load_network(folder: str | Path) -> Network:
    """Read ``dcs.csv``, ``customers.csv``, ``products.csv``, ``demand.csv`` and ``baseline.csv`` from a folder and validate them."""
    folder = Path(folder)
    missing = [f"{name}.csv" for name in TABLES if not (folder / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing: {', '.join(missing)}")
    text = {"dc": str, "customer_id": str, "week": str, "product_family": str, "mode": str, "city": str, "state": str}
    return validate_network(*(pd.read_csv(folder / f"{name}.csv", dtype=text, keep_default_na=False, na_values=[""])
                              for name in TABLES))


# --------------------------------------------------------------- sample data
_SAMPLE_DCS = {
    # city: (handling cost per unit, fixed cost per week, share of customers that still ship from it out of habit)
    "Los Angeles": (0.62, 9500.0, 0.0),
    "Dallas": (0.44, 7000.0, 0.0),
    "Chicago": (0.52, 9000.0, 0.5),
    "Atlanta": (0.46, 7500.0, 0.0),
    "Newark": (0.68, 10500.0, 0.0),
    "Memphis": (0.40, 8000.0, 0.5),
}
_SAMPLE_PRODUCTS = {"Accessories": 1.5, "Consumables": 9.0, "Hardware": 28.0, "Equipment": 85.0}
_SEGMENTS = {
    # segment: (share of customers, median pounds per week, spread, shipments-per-week choices, weight mix by family)
    "small": (0.40, 180.0, 0.60, (2, 3, 5), (0.35, 0.45, 0.15, 0.05)),
    "medium": (0.45, 2500.0, 0.70, (1, 2, 3), (0.10, 0.35, 0.35, 0.20)),
    "large": (0.15, 22000.0, 0.50, (1, 2, 3), (0.05, 0.25, 0.35, 0.35)),
}


def generate_sample_network(customers: int = 150, weeks: int = 13, seed: int = 7) -> Network:
    """A simulated US distribution network, for demos and tests. None of it is real.

    Six distribution centers in real cities serve ``customers`` locations in
    US metro areas for ``weeks`` weeks. Customers come in three sizes (small
    parcel-sized accounts, medium accounts, large accounts) and order four
    product families of different weight per unit. Coordinates are
    approximate; capacities, costs and demand are invented.

    The "as-is" allocation is built to look like habit rather than choice:
    about half the customers ship from their nearest DC, about a fifth from
    the second nearest, and the rest from one of two older sites (Chicago and
    Memphis) wherever the customer is. Modes follow a rule of thumb on
    shipment weight, and some customers use a heavier mode than they need.
    Each DC's capacity is its busiest week under that allocation plus 25%,
    rounded up to the next 500 units, so today's allocation always fits.
    """
    if customers < 1 or customers > len(ALL_METROS):
        raise ValueError(f"customers must be between 1 and {len(ALL_METROS)}")
    if weeks < 1:
        raise ValueError("weeks must be at least 1")
    rng = np.random.default_rng(seed)

    metros = sorted(ALL_METROS)
    chosen = sorted(rng.choice(len(metros), customers, replace=False))
    names = [metros[i] for i in chosen]
    segments = list(_SEGMENTS)
    segment = rng.choice(segments, customers, p=[_SEGMENTS[s][0] for s in segments])
    base = np.array([_SEGMENTS[s][1] * rng.lognormal(0, _SEGMENTS[s][2]) for s in segment])
    frequency = np.array([rng.choice(_SEGMENTS[s][3]) for s in segment])
    cust = pd.DataFrame({
        "customer_id": [f"C{i + 1:03d}" for i in range(customers)],
        "city": names,
        "state": [ALL_METROS[m][0] for m in names],
        "latitude": [ALL_METROS[m][1] for m in names],
        "longitude": [ALL_METROS[m][2] for m in names],
        "segment": segment,
        "shipments_per_week": frequency,
    })

    families = list(_SAMPLE_PRODUCTS)
    unit_weight = np.array([_SAMPLE_PRODUCTS[f] for f in families])
    mix = np.vstack([rng.dirichlet(np.array(_SEGMENTS[s][4]) * 8.0) for s in segment])              # customer x family
    season = 1.0 + 0.10 * np.sin(2 * np.pi * np.arange(weeks) / max(weeks, 2))                       # a mild swell
    level = base[:, None] * season[None, :] * rng.lognormal(0, 0.22, (customers, weeks))            # pounds per week
    units = np.round(level[:, :, None] * mix[:, None, :] * rng.lognormal(0, 0.10, (customers, weeks, len(families)))
                     / unit_weight[None, None, :])
    units[:, :, 1] = np.maximum(units[:, :, 1], 1)                                                   # every customer orders every week
    week_labels = pd.date_range("2026-01-05", periods=weeks, freq="7D").strftime("%Y-%m-%d")
    ci, wi, fi = np.meshgrid(np.arange(customers), np.arange(weeks), np.arange(len(families)), indexing="ij")
    demand = pd.DataFrame({"week": week_labels[wi.ravel()], "customer_id": cust["customer_id"].to_numpy()[ci.ravel()],
                           "product_family": np.array(families)[fi.ravel()], "units": units.ravel()})
    demand = demand[demand["units"] > 0].reset_index(drop=True)
    products = pd.DataFrame({"product_family": families, "weight_lbs_per_unit": unit_weight})

    dc_names = list(_SAMPLE_DCS)
    dcs = pd.DataFrame({
        "dc": dc_names,
        "city": dc_names,
        "state": [ALL_METROS[m][0] for m in dc_names],
        "latitude": [ALL_METROS[m][1] for m in dc_names],
        "longitude": [ALL_METROS[m][2] for m in dc_names],
        "handling_cost_per_unit": [_SAMPLE_DCS[m][0] for m in dc_names],
        "fixed_cost_per_week": [_SAMPLE_DCS[m][1] for m in dc_names],
    })

    # Today's allocation: habit, not optimization.
    miles = road_miles(dcs["latitude"].to_numpy()[:, None], dcs["longitude"].to_numpy()[:, None],
                       cust["latitude"].to_numpy()[None, :], cust["longitude"].to_numpy()[None, :])
    order = np.argsort(miles, axis=0)                                                                # nearest DC first
    habit = rng.choice(3, customers, p=[0.55, 0.22, 0.23])
    legacy = rng.choice(len(dc_names), customers, p=[_SAMPLE_DCS[m][2] for m in dc_names])
    served_by = np.where(habit == 0, order[0], np.where(habit == 1, order[1], legacy))
    typical = (units * unit_weight[None, None, :]).sum(axis=2).mean(axis=1) / frequency               # pounds per shipment
    step = np.select([typical <= 200, typical < 12000], [0, 1], 2)
    heavier = (rng.random(customers) < 0.12) & (step < 2) & ((step == 0) | (typical > 4000))
    mode = np.array(MODES)[step + heavier]
    baseline = pd.DataFrame({"customer_id": cust["customer_id"], "dc": np.array(dc_names)[served_by], "mode": mode, "share": 1.0})

    weekly_units = units.sum(axis=2)
    peak = np.array([weekly_units[served_by == i].sum(axis=0).max() if (served_by == i).any() else 0.0 for i in range(len(dc_names))])
    dcs.insert(5, "capacity_units_per_week", np.maximum(500.0, np.ceil(peak * 1.25 / 500.0) * 500.0))
    return validate_network(dcs, cust, products, demand, baseline)
