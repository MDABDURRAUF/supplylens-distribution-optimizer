from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import distribution_optimizer as do
from distribution_optimizer.cli import main
from distribution_optimizer.us import MORE_METROS, STATE_REGION

# Simple rates for hand-worked cases: truckload only, $1 a mile per truck, no minimum, no fuel surcharge.
PLAIN = do.RateCard(fuel_surcharge=0.0, tl_per_mile=1.0, tl_min=0.0)
TRUCK = do.Settings(max_transit_days=None, modes=("truckload",), rates=PLAIN)


@pytest.fixture(scope="module")
def network():
    return do.generate_sample_network()


@pytest.fixture(scope="module")
def plans(network):
    settings = do.Settings()
    baseline, plan = do.baseline_plan(network, settings), do.optimize(network, settings)
    return settings, baseline, plan, do.compare(baseline, plan)


def tiny(dcs, customers, weeks=1, weight=1.0, baseline=None):
    """A hand-made network.

    dcs: (name, latitude, longitude, capacity, handling cost per unit, fixed cost per week)
    customers: (id, latitude, longitude, units per week, shipments per week)
    Every customer is served today by the first DC unless ``baseline`` says otherwise.
    """
    d = pd.DataFrame(dcs, columns=["dc", "latitude", "longitude", "capacity_units_per_week", "handling_cost_per_unit", "fixed_cost_per_week"])
    c = pd.DataFrame(customers, columns=["customer_id", "latitude", "longitude", "units", "shipments_per_week"])
    d["state"], c["state"] = "IL", "IL"
    demand = pd.DataFrame([{"week": f"2026-01-{5 + 7 * w:02d}", "customer_id": r.customer_id, "product_family": "item", "units": r.units}
                           for w in range(weeks) for r in c.itertuples()])
    base = pd.DataFrame(baseline or [(r.customer_id, d["dc"].iloc[0]) for r in c.itertuples()], columns=["customer_id", "dc"])
    return do.validate_network(d, c.drop(columns="units"), pd.DataFrame({"product_family": ["item"], "weight_lbs_per_unit": [weight]}), demand, base)


# ---------------------------------------------------------------- reference
def test_us_reference_data():
    assert do.great_circle_miles(41.88, -87.63, 32.78, -96.80) == pytest.approx(803, abs=15)  # Chicago to Dallas
    assert do.road_miles(41.88, -87.63, 32.78, -96.80) == pytest.approx(1.18 * do.great_circle_miles(41.88, -87.63, 32.78, -96.80))
    assert do.region_of("il") == "Midwest" and len({s for g in do.CENSUS_REGIONS.values() for s in g}) == 51
    assert len(do.ALL_METROS) == len(do.METROS) + len(MORE_METROS) >= 150  # no name is used twice
    for state, lat, lon in do.ALL_METROS.values():  # the contiguous United States only
        assert state in STATE_REGION and state not in ("AK", "HI") and 24 < lat < 50 and -125 < lon < -66


# -------------------------------------------------------------------- rates
def test_rates_match_hand_worked_charges():
    cost = lambda mode, miles, lbs: float(do.shipment_cost(mode, miles, lbs))  # noqa: E731  (default card, 20% fuel)
    assert cost("parcel", 500, 10) == pytest.approx((9 + 10 * (0.28 + 0.00045 * 500)) * 1.2)       # 16.86
    assert cost("parcel", 0, 1) == pytest.approx(12 * 1.2)                                           # minimum charge
    assert cost("parcel", 500, 300) == pytest.approx(2 * cost("parcel", 500, 150))                   # two packages
    assert cost("LTL", 1000, 1000) == pytest.approx(10 * (14 + 35) * 0.76 * 1.2)                     # 446.88
    assert cost("LTL", 1000, 900) == pytest.approx(cost("LTL", 1000, 1000))                          # billed at the next break
    assert cost("LTL", 1000, 400) == pytest.approx(4 * 49 * 1.00 * 1.2)                              # own break is cheaper
    assert cost("LTL", 100, 50) == pytest.approx(110 * 1.2)                                          # minimum charge
    assert cost("truckload", 1000, 30000) == pytest.approx(2300 * 1.2) == cost("truckload", 1000, 5)
    assert cost("truckload", 100, 30000) == pytest.approx(650 * 1.2)                                 # minimum charge
    assert cost("truckload", 1000, 50000) == pytest.approx(2 * 2300 * 1.2)                           # two trucks
    assert float(do.shipment_count("truckload", 50000)) == 2 and float(do.shipment_count("parcel", 0)) == 0
    assert cost("LTL", 800, 0) == 0
    assert float(do.shipment_cost("LTL", 700, 2500, do.RateCard(fuel_surcharge=0.5))) == pytest.approx(
        1.5 * float(do.shipment_cost("LTL", 700, 2500, do.RateCard(fuel_surcharge=0.0))))
    weights = np.arange(1, 60000, 37.0)
    for mode, curve in do.rate_curve(900, weights).items():  # a heavier shipment never costs less
        assert (np.diff(curve) >= -1e-9).all(), mode
    with pytest.raises(ValueError, match="not a known mode"):
        do.shipment_cost("rail", 100, 100)
    with pytest.raises(ValueError, match="ltl_breaks"):
        do.RateCard(ltl_breaks=((100, 1.0),))


def test_transit_days_by_mode_and_distance():
    assert do.transit_days("parcel", [100, 400, 401, 900]).tolist() == [1, 1, 2, 3]
    assert do.transit_days("LTL", [100, 450, 451, 900]).tolist() == [2, 2, 3, 3]          # one handling day
    assert do.transit_days("truckload", [100, 500, 501, 1600]).tolist() == [1, 1, 2, 4]


# --------------------------------------------------------------------- data
def test_sample_network_is_reproducible_and_sensible(network):
    again = do.generate_sample_network()
    for name in ("dcs", "customers", "products", "demand", "baseline"):
        pd.testing.assert_frame_equal(getattr(network, name), getattr(again, name))
    assert len(network.dcs) == 6 and len(network.customers) == 150 and len(network.weeks) == 13 and len(network.products) == 4
    assert set(network.customers["region"]) == {"Northeast", "Midwest", "South", "West"}
    assert network.customers["city"].is_unique and set(network.baseline["mode"]) == set(do.MODES)
    units, weight = network.weekly()
    assert units.shape == (150, 13) and (units.sum(axis=1) > 0).all() and (weight > 0).all()
    baseline = do.baseline_plan(network, do.Settings(max_transit_days=None))
    assert baseline.violations == []                                  # today's allocation fits every DC
    assert baseline.dc_summary["peak_utilization"].between(0.5, 0.8).all()
    # Habit, not choice: fewer than two in three customers ship from their nearest DC today.
    nearest = np.array(network.dcs["dc"])[network.distances().argmin(axis=0)]
    assert 0.4 < (network.baseline["dc"].to_numpy() == nearest).mean() < 0.67
    assert len(do.generate_sample_network(customers=20, weeks=3, seed=1).demand) > 0


def test_validation_reports_problems(network):
    tables = [getattr(network, n).copy() for n in ("dcs", "customers", "products", "demand", "baseline")]
    with pytest.raises(do.SchemaError, match="missing required column.*capacity_units_per_week"):
        do.validate_network(tables[0].drop(columns="capacity_units_per_week"), *tables[1:])
    bad = [t.copy() for t in tables]
    bad[1].loc[0, "state"] = "ZZ"
    bad[3].loc[:2, "units"] = -5
    bad[3].loc[5, "customer_id"] = "nobody"
    bad[4].loc[1, "dc"] = "Atlantis"
    bad[4].loc[2, "mode"] = "rail"
    with pytest.raises(do.SchemaError) as err:
        do.validate_network(*bad)
    text = str(err.value)
    for part in ("customers: 1 row(s) with a state code", "demand: 3 row(s) with a negative 'units'", "demand: 1 row(s) whose 'customer_id'",
                 "baseline: 1 row(s) whose 'dc'", "baseline: 1 row(s) with a mode other than"):
        assert part in text
    short = [t.copy() for t in tables]
    short[4] = short[4].iloc[1:]
    with pytest.raises(do.SchemaError, match="1 customer.s. with no row saying how they are served today"):
        do.validate_network(*short)
    idle = [t.copy() for t in tables]
    idle[3] = idle[3][idle[3]["customer_id"] != "C001"]
    with pytest.raises(do.SchemaError, match="1 customer.s. with no demand"):
        do.validate_network(*idle)


def test_network_files_round_trip(network, tmp_path):
    folder = network.save(tmp_path / "net")
    loaded = do.load_network(folder)
    for name in ("dcs", "customers", "products", "demand", "baseline"):
        pd.testing.assert_frame_equal(getattr(network, name), getattr(loaded, name), check_dtype=False)
    (folder / "demand.csv").unlink()
    with pytest.raises(FileNotFoundError, match="demand.csv"):
        do.load_network(folder)


# ----------------------------------------------------- hand-worked optimums
def test_capacity_and_single_sourcing_by_hand():
    # Two DCs at the same place, so freight is the same from either and only handling differs: A $1, B $3 a unit.
    # A is cheaper but holds 35 units; customers need 10, 20 and 30.
    net = tiny([("A", 40, -90, 35, 1.0, 0), ("B", 40, -90, 100, 3.0, 0)],
               [("c1", 41, -90, 10, 1), ("c2", 41, -90, 20, 1), ("c3", 41, -90, 30, 1)], baseline=[("c1", "B"), ("c2", "B"), ("c3", "B")])
    split = do.optimize(net, TRUCK)
    # Splitting allowed: fill A with 35 units, 25 go through B.  35 * 1 + 25 * 3 = 110.
    assert split.status == "optimal" and split.handling_cost == pytest.approx(110) and split.split_customers >= 1
    assert split.dc_summary.set_index("dc")["peak_week_units"].to_dict() == pytest.approx({"A": 35, "B": 25})
    # One DC per customer: A can take 30 units at most (c3, or c1 and c2).  30 * 1 + 30 * 3 = 120.
    single = do.optimize(net, replace(TRUCK, single_sourcing=True))
    assert single.handling_cost == pytest.approx(120) and single.split_customers == 0
    assert single.dc_summary.set_index("dc")["units"].to_dict() == pytest.approx({"A": 30, "B": 30})
    assert split.freight_cost == pytest.approx(single.freight_cost)   # same miles either way
    # Today everything ships from B: 60 * 3 = 180, so the estimated saving is 70 and 60.
    base = do.baseline_plan(net, TRUCK)
    assert base.handling_cost == pytest.approx(180)
    assert do.compare(base, split).saving == pytest.approx(70) and do.compare(base, single).saving == pytest.approx(60)
    # Using only 80% of A's capacity (28 units): 28 * 1 + 32 * 3 = 124.
    assert do.optimize(net, replace(TRUCK, max_capacity_use=0.8)).handling_cost == pytest.approx(124)


def test_nearest_dc_and_freight_by_hand():
    # One truck a week at $1 a mile: freight is simply road miles, so each customer goes to its nearer DC.
    net = tiny([("West", 40, -100, 100, 0.0, 0), ("East", 40, -80, 100, 0.0, 0)],
               [("w", 40, -98, 10, 1), ("e", 40, -83, 10, 1)], weeks=2, baseline=[("w", "East"), ("e", "West")])
    miles = pd.DataFrame(net.distances(), index=["West", "East"], columns=["w", "e"])
    assert miles.loc["West", "w"] == pytest.approx(do.road_miles(40, -100, 40, -98)) and miles.loc["West", "w"] < miles.loc["East", "w"]
    plan = do.optimize(net, TRUCK)
    assert plan.assignment.set_index("customer_id")["dc"].to_dict() == {"w": "West", "e": "East"}
    assert plan.freight_cost == pytest.approx(2 * (miles.loc["West", "w"] + miles.loc["East", "e"]))       # two weeks
    base = do.baseline_plan(net, TRUCK)
    assert base.freight_cost == pytest.approx(2 * (miles.loc["East", "w"] + miles.loc["West", "e"]))
    c = do.compare(base, plan)
    assert c.saving == pytest.approx(base.freight_cost - plan.freight_cost) and c.customers_changed == 2 and c.dc_changes == 2
    assert c.lane_changes["saving"].sum() == pytest.approx(c.saving) and set(c.lane_changes["change"]) == {"DC"}
    # Three shipments a week means three trucks a week.
    thrice = tiny([("West", 40, -100, 100, 0.0, 0)], [("w", 40, -98, 10, 3)], weeks=2)
    assert do.optimize(thrice, TRUCK).freight_cost == pytest.approx(3 * 2 * miles.loc["West", "w"])
    assert do.optimize(thrice, TRUCK).lanes["shipments"].iloc[0] == 6


def test_open_or_close_a_dc_by_hand():
    # Two DCs at the same place. A: handling $1, fixed $50 a week. B: handling $2, fixed $10 a week. 60 units a week.
    def net(fixed_a):
        return tiny([("A", 40, -90, 100, 1.0, fixed_a), ("B", 40, -90, 100, 2.0, 10)], [("c1", 41, -90, 60, 1)])
    keep = do.optimize(net(50), TRUCK)                               # both stay open: A serves all. 60 + 50 + 10 = 120
    assert keep.open_dcs == ["A", "B"] and keep.handling_cost + keep.fixed_cost == pytest.approx(120)
    choose = replace(TRUCK, choose_dcs=True)
    a_only = do.optimize(net(50), choose)                            # A alone: 60 + 50 = 110; B alone: 120 + 10 = 130
    assert a_only.open_dcs == ["A"] and a_only.handling_cost + a_only.fixed_cost == pytest.approx(110)
    b_only = do.optimize(net(100), choose)                           # A alone: 60 + 100 = 160; B alone: 130
    assert b_only.open_dcs == ["B"] and b_only.handling_cost + b_only.fixed_cost == pytest.approx(130)
    out = do.open_close(net(100), TRUCK)
    assert out["closed"] == ["A"] and out["saving"] == pytest.approx((60 + 100 + 10) - 130)
    table = do.dc_count_table(net(100), TRUCK)
    assert table["open"].tolist() == ["B", "A, B"] and table["fixed_cost"].tolist() == [10, 110]
    # A closed DC cannot be used: capacity of the open one must hold the demand.
    tight = tiny([("A", 40, -90, 100, 1.0, 100), ("B", 40, -90, 50, 2.0, 10)], [("c1", 41, -90, 60, 1)])
    assert do.optimize(tight, choose).open_dcs == ["A"]
    assert do.optimize(tight, TRUCK, n_open=2).open_dcs == ["A", "B"]


def test_mode_choice_and_service_rule_by_hand():
    # One DC, one customer about 1,000 road miles away, one shipment a week.
    def lane(lbs, **settings):
        net = tiny([("A", 40, -100, 1e6, 0.0, 0)], [("c", 40, -84, lbs, 1)])
        return net, do.optimize(net, do.Settings(**settings))
    net, light = lane(40)
    miles = float(net.distances()[0, 0])
    assert 950 < miles < 1050
    by_mode = {m: float(do.shipment_cost(m, miles, 40)) for m in do.MODES}
    assert light.lanes["mode"].iloc[0] == "parcel" and light.freight_cost == pytest.approx(by_mode["parcel"]) == min(by_mode.values())
    assert lane(2000)[1].lanes["mode"].iloc[0] == "LTL" and lane(30000)[1].lanes["mode"].iloc[0] == "truckload"
    # LTL takes 1 + ceil(miles / 450) = 4 days, parcel ceil(miles / 400) = 3, truckload ceil(miles / 500) = 2.
    assert lane(2000, max_transit_days=4)[1].lanes["transit_days"].iloc[0] == 4
    faster = lane(2000, max_transit_days=3)[1]                       # LTL is too slow: 14 parcels or a truck
    options = {m: float(do.shipment_cost(m, miles, 2000)) for m in ("parcel", "truckload")}
    assert faster.lanes["mode"].iloc[0] == min(options, key=options.get) and faster.freight_cost == pytest.approx(min(options.values()))
    assert faster.freight_cost > lane(2000)[1].freight_cost and faster.share_within_service == 1.0
    assert lane(2000, max_transit_days=2)[1].lanes["mode"].iloc[0] == "truckload"
    net, none = lane(2000, max_transit_days=1)
    assert none.status == "infeasible" and "cannot be reached within 1 day" in none.message and "c (" in none.message
    assert do.diagnose(net, do.Settings(max_transit_days=1)) == ["the service rule (a plan exists at 2 transit days)"]
    assert do.minimum_transit_days(net, do.Settings(max_transit_days=1, modes=("LTL",))) == 4
    # Today's lane is costed with today's mode and judged against the rule.
    base = do.cost_plan(net, pd.DataFrame({"customer_id": ["c"], "dc": ["A"], "mode": ["LTL"]}), do.Settings(max_transit_days=3))
    assert base.share_within_service == 0.0 and "slower than the 3-day service rule" in base.violations[0]
    assert base.freight_cost == pytest.approx(float(do.shipment_cost("LTL", miles, 2000)))


def test_infeasible_plans_name_the_rule_to_relax():
    net = tiny([("A", 40, -90, 30, 1.0, 0), ("B", 40, -90, 20, 3.0, 0)], [("c1", 41, -90, 25, 1), ("c2", 41, -90, 25, 1)])
    assert do.optimize(net, TRUCK).status == "optimal"               # 50 units fit 50 of capacity when customers are split
    single = replace(TRUCK, single_sourcing=True)
    assert do.optimize(net, single).status == "infeasible"           # but no DC can take two whole customers, and B not even one
    assert do.diagnose(net, single) == ["DC capacity", "single sourcing"]
    assert do.diagnose(net, TRUCK) == []
    grown = net.with_demand_growth(0.2)                              # 60 units against 50 of capacity
    plan = do.optimize(grown, TRUCK)
    assert plan.status == "infeasible" and "60 units" in plan.message and "50 units" in plan.message
    assert do.diagnose(grown, TRUCK) == ["DC capacity"]
    table = do.demand_growth(net, TRUCK, growths=(0.0, 0.2))
    assert table["status"].tolist() == ["optimal", "infeasible"] and "DC capacity" in table["note"].iloc[1]
    assert table["peak_utilization"].iloc[0] == pytest.approx(1.0) and table["dcs_full"].iloc[0] == 2
    # About 1,000 miles by LTL only takes 4 days; a truck would make the 2-day rule.
    far = tiny([("A", 40, -100, 5000, 0.0, 0)], [("c", 40, -84, 1000, 1)])
    slow = do.Settings(max_transit_days=2, modes=("LTL",))
    assert do.diagnose(far, slow) == ["the service rule (a plan exists at 4 transit days)", "the allowed modes"]
    # Two rules in the way at once: no single rule is to blame.
    assert do.diagnose(far.with_demand_growth(9.0), slow) == []


# ------------------------------------------------------------- costing plans
def test_cost_plan_is_like_for_like(network, plans):
    settings, baseline, plan, _ = plans
    again = do.cost_plan(network, plan.assignment, settings)
    assert again.total_cost == pytest.approx(plan.total_cost) and again.violations == []
    assert again.lanes["mode"].tolist() == plan.lanes["mode"].tolist()
    for r in (baseline, plan):
        assert r.total_cost == pytest.approx(r.freight_cost + r.handling_cost + r.fixed_cost)
        assert r.lanes["freight_cost"].sum() == pytest.approx(r.freight_cost) and r.units == pytest.approx(network.demand["units"].sum())
        assert r.fixed_cost == pytest.approx(13 * network.dcs["fixed_cost_per_week"].sum())
    # A lane that carries 40% of a customer costs 40% of serving the whole customer from that DC.
    whole = do.cost_plan(network, pd.DataFrame({"customer_id": ["C001"], "dc": ["Dallas"], "mode": ["LTL"]}), settings)
    part = do.cost_plan(network, pd.DataFrame({"customer_id": ["C001", "C001"], "dc": ["Dallas", "Newark"], "share": [0.4, 0.6],
                                               "mode": ["LTL", "LTL"]}), settings)
    lane = part.lanes.set_index("dc").loc["Dallas"]
    assert lane["freight_cost"] == pytest.approx(0.4 * whole.freight_cost) and lane["handling_cost"] == pytest.approx(0.4 * whole.handling_cost)
    assert "149 customer(s) are not fully served" in whole.violations[0]
    # Problems with a plan are reported, not hidden.
    crowded = network.baseline.assign(dc="Newark")
    assert any(v.startswith("Newark: busiest week ships") for v in do.cost_plan(network, crowded, settings).violations)
    gone = do.cost_plan(network.without_dc("Memphis"), network.baseline, settings)
    assert any("not in the network: Memphis" in v for v in gone.violations)
    with pytest.raises(ValueError, match="unknown mode"):
        do.cost_plan(network, network.baseline.assign(mode="rail"), settings)
    options = do.lane_options(network, settings)
    assert len(options) == 6 * 150 * 3 and options.groupby(["customer_id", "dc"])["cheapest"].sum().max() == 1
    best = options[options["cheapest"]].set_index(["customer_id", "dc"])["mode"]
    assert (plan.lanes.set_index(["customer_id", "dc"])["mode"] == best.reindex(plan.lanes.set_index(["customer_id", "dc"]).index)).all()


def test_sample_optimization_obeys_every_rule(network, plans):
    settings, baseline, plan, c = plans
    assert plan.status == "optimal" and plan.violations == []
    assert plan.lanes.groupby("customer_id")["share"].sum().to_numpy() == pytest.approx(1.0)
    assert plan.lanes["within_service"].all() and plan.share_within_service == 1.0 and baseline.share_within_service < 1.0
    assert (plan.dc_summary["peak_week_units"] <= plan.dc_summary["capacity_units_per_week"] + 1e-6).all()
    assert 0.05 < c.saving_pct < 0.30 and c.saving == pytest.approx(baseline.total_cost - plan.total_cost)
    assert c.costs.set_index("component").loc["Fixed DC cost", "change"] == 0   # every DC stays open in both
    assert c.customers == 150 and c.customers_changed == len(c.lane_changes) and c.saving_per_week == pytest.approx(c.saving / 13)
    assert plan.average_transit_days < baseline.average_transit_days
    # Tighter rules can only cost more; more freedom can only cost less.
    single = do.optimize(network, replace(settings, single_sourcing=True))
    free = do.optimize(network, replace(settings, max_transit_days=None))
    choose = do.optimize(network, replace(settings, choose_dcs=True))
    assert single.status == "optimal" and single.split_customers == 0 and single.total_cost >= plan.total_cost - 1
    assert free.total_cost <= plan.total_cost + 1 and choose.total_cost <= plan.total_cost + 1
    assert len(choose.open_dcs) < 6 and choose.dc_summary.loc[~choose.dc_summary["open"], ["units", "fixed_cost"]].to_numpy().sum() == 0
    # Mode choice alone, with nobody moving, explains part of the saving and never adds cost.
    modes_only = do.baseline_with_best_modes(network, settings)
    assert plan.total_cost < modes_only.total_cost < baseline.total_cost
    assert (modes_only.lanes.set_index("customer_id")["dc"] == baseline.lanes.set_index("customer_id")["dc"]).all()


# ------------------------------------------------------------- consolidation
def test_consolidation_by_hand():
    # Five 100 lb LTL shipments a week to a customer next door; each pays the $110 minimum (no fuel surcharge).
    net = tiny([("A", 40, -90, 1e6, 0.0, 0)], [("c", 40, -90, 500, 5)])
    def run(rule, hold):
        settings = do.Settings(max_transit_days=rule, modes=("LTL",), rates=do.RateCard(fuel_surcharge=0.0))
        return do.consolidation_analysis(net, do.optimize(net, settings), settings, max_hold_days=hold).iloc[0]
    assert do.shipments_after_hold([5, 5, 5, 5, 2, 1], 0).tolist() == [5, 5, 5, 5, 2, 1]
    assert [int(do.shipments_after_hold(5, h)) for h in (1, 2, 3, 4)] == [3, 2, 2, 1] and int(do.shipments_after_hold(2, 1)) == 2
    none = run(None, 0)
    assert none["freight_before"] == pytest.approx(550) and none["saving"] == 0 and none["hold_days"] == 0
    # No service rule: hold four days and send one 500 lb shipment, still at the minimum.  550 -> 110.
    full = run(None, 4)
    assert full["hold_days"] == 4 and full["consolidated_shipments_per_week"] == 1 and full["freight_after"] == pytest.approx(110)
    assert full["lbs_per_shipment"] == pytest.approx(100) and full["consolidated_lbs_per_shipment"] == pytest.approx(500)
    # A local LTL shipment takes 2 days. With a 4-day rule only a 2-day hold fits: two shipments a week.  550 -> 220.
    two = run(4, 4)
    assert two["hold_days"] == 2 and two["freight_after"] == pytest.approx(220) and two["order_to_delivery_days"] == 4
    assert run(4, 1)["freight_after"] == pytest.approx(330) and run(2, 4)["hold_days"] == 0


def test_consolidation_on_the_sample(network, plans):
    settings, _, plan, _ = plans
    table = do.consolidation_analysis(network, plan, settings, max_hold_days=2)
    s = do.consolidation_summary(table)
    assert len(table) == len(plan.lanes) and s["freight_before"] == pytest.approx(plan.freight_cost)
    assert (table["saving"] >= 0).all() and s["saving"] > 0 and 0 < s["lanes_consolidated"] < s["lanes"]
    assert s["shipments_per_week_before"] == network.customers["shipments_per_week"].sum() > s["shipments_per_week_after"]
    assert (table["order_to_delivery_days"] <= settings.max_transit_days).all()
    assert (table.loc[table["hold_days"] == 0, "saving"] == 0).all()
    windows = do.consolidation_by_window(network, plan, settings, windows=(0, 1, 2))
    assert windows["saving"].iloc[0] == 0 and windows["saving"].is_monotonic_increasing


# ----------------------------------------------------------------- scenarios
def test_fuel_sensitivity(network, plans):
    settings, baseline, plan, _ = plans
    table = do.fuel_sensitivity(network, settings, surcharges=(0.0, 0.2, 0.4))
    now = table[table["fuel_surcharge"] == 0.2].iloc[0]
    assert now["baseline_cost"] == pytest.approx(baseline.total_cost) and now["reoptimized_cost"] == pytest.approx(plan.total_cost)
    assert now["keep_plan_cost"] == pytest.approx(plan.total_cost) and now["customers_moved"] == 0
    assert table["baseline_cost"].is_monotonic_increasing and table["reoptimized_cost"].is_monotonic_increasing
    assert (table["reoptimizing_gain"] >= -1e-6).all() and (table["saving_vs_baseline"] > 0).all()
    # The surcharge scales freight only: 20 points on today's allocation is 20/120 of its freight.
    assert table["baseline_cost"].diff().iloc[1] == pytest.approx(baseline.freight_cost / 1.2 * 0.2)


def test_dc_loss_and_growth_on_the_sample(network, plans):
    settings, _, plan, _ = plans
    table = do.dc_loss_table(network, settings).set_index("dc_lost")
    assert set(table.index) == set(network.dcs["dc"]) and set(table["outcome"]) == {"absorbed", "no feasible plan"}
    assert table.loc["Chicago", "outcome"] == "no feasible plan" and table.loc["Chicago", "relax"] == "DC capacity"
    absorbed = table[table["outcome"] == "absorbed"]
    assert (absorbed["extra_cost"] > 0).all() and (absorbed["share_within_service"] == 1.0).all()
    one = do.dc_loss(network, settings, "Memphis", plan)
    assert "Memphis" not in set(one["adapt"].lanes["dc"]) and one["extra_cost"] == pytest.approx(table.loc["Memphis", "extra_cost"])
    assert one["extra_cost"] == pytest.approx(one["adapt"].total_cost - (plan.total_cost - 13 * 8000.0))
    assert one["units_affected"] == pytest.approx(plan.lanes.loc[plan.lanes["dc"] == "Memphis", "units"].sum())
    growth = do.demand_growth(network, settings, growths=(0.0, 0.2, 0.6))
    assert growth["status"].tolist() == ["optimal", "optimal", "infeasible"] and growth["total_cost"].iloc[0] == pytest.approx(plan.total_cost)
    assert growth["cost_per_unit"].iloc[1] < growth["cost_per_unit"].iloc[0]     # fixed cost is spread over more units
    assert "capacity" in growth["note"].iloc[2]
    with pytest.raises(ValueError, match="not a DC"):
        network.without_dc("Atlantis")


def test_open_close_on_the_sample(network, plans):
    settings, _, plan, _ = plans
    out = do.open_close(network, settings)
    assert out["all_open"].total_cost == pytest.approx(plan.total_cost) and out["saving"] > 0 and len(out["closed"]) >= 1
    table = do.dc_count_table(network, settings)
    feasible = table[table["status"] == "optimal"]
    assert table["dcs_open"].tolist() == [1, 2, 3, 4, 5, 6] and table["status"].iloc[0] == "infeasible"   # one DC cannot hold it all
    assert feasible["total_cost"].min() == pytest.approx(out["chosen"].total_cost)
    assert feasible["fixed_cost"].is_monotonic_increasing and feasible["freight_cost"].is_monotonic_decreasing


# ------------------------------------------------------------ report and CLI
def test_report_and_headline(network, plans):
    settings, baseline, plan, c = plans
    text = do.savings_headline(c, baseline, plan, settings)
    assert text.startswith("Estimated saving: $") and "not a realized saving" in text and f"{c.customers_changed} of 150 customers" in text
    html = do.build_report(baseline, plan, c, settings, consolidation=do.consolidation_analysis(network, plan, settings),
                           dc_loss=do.dc_loss_table(network, settings), fuel=do.fuel_sensitivity(network, settings, (0.1, 0.2)), simulated=True)
    for part in ("<!doctype html>", "Cost against today", "Lanes that change", "Consolidation", "If a DC is lost", "Fuel surcharge",
                 "simulated", "not real tariffs", "not a realized saving"):
        assert part in html
    assert "<script" not in html and "http" not in html               # one self-contained file
    worse = do.compare(plan, baseline)                                 # the other way round is reported as a cost, not a saving
    assert worse.saving < 0 and do.savings_headline(worse, plan, baseline, settings).startswith("No cheaper allocation")
    assert "delivery within 4 transit days" in do.rules_text(settings) and "no service rule" in do.rules_text(do.Settings(max_transit_days=None))


def test_command_line(tmp_path, capsys):
    data = tmp_path / "net"
    assert main(["sample", "--out", str(data), "--customers", "40", "--weeks", "4", "--seed", "3"]) == 0
    assert sorted(p.name for p in data.iterdir()) == ["baseline.csv", "customers.csv", "dcs.csv", "demand.csv", "products.csv"]
    out, changes, report = tmp_path / "assignment.csv", tmp_path / "changes.csv", tmp_path / "report.html"
    assert main(["optimize", "--data", str(data), "--out", str(out), "--changes", str(changes), "--single-sourcing"]) == 0
    printed = capsys.readouterr().out
    assert "Estimated saving" in printed and "not a realized saving" in printed
    assignment = pd.read_csv(out)
    assert len(assignment) == 40 and assignment["share"].eq(1).all() and {"dc", "mode", "freight_cost"} <= set(assignment.columns)
    assert {"from_dc", "to_dc", "saving"} <= set(pd.read_csv(changes).columns)
    for kind in ("fuel", "growth", "dc-loss", "open-close", "consolidation"):
        table = tmp_path / f"{kind}.csv"
        assert main(["scenario", kind, "--data", str(data), "--out", str(table)]) == 0 and len(pd.read_csv(table)) > 0
    assert main(["report", "--data", str(data), "--out", str(report), "--choose-dcs"]) == 0
    assert "Distribution optimization report" in report.read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["optimize", "--data", str(data), "--max-transit-days", "1"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "the service rule" in err
    assert main(["optimize", "--data", str(tmp_path / "missing")]) == 1
