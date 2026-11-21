"""SupplyLens Distribution Optimizer dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import distribution_optimizer as do
from distribution_optimizer.data import TABLES
from distribution_optimizer.report import money, pct

st.set_page_config(page_title="Distribution Optimizer", layout="wide")

RED, AMBER, GREEN, BLUE, GREY, DARK = "#d1495b", "#e9a23b", "#2a9d8f", "#3a6ea5", "#9aa5b1", "#1c2430"
PALETTE = [BLUE, GREEN, AMBER, RED, "#7b5ea7", "#4f8a8b", "#c26a3d", "#5c6b7a"]
MODE_COLORS = {"parcel": AMBER, "LTL": BLUE, "truckload": GREEN}
PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="dollar")
NUM = st.column_config.NumberColumn(format="localized")
SOLVER_SECONDS = 5.0   # the longest one plan may take in the dashboard


def show(fig: go.Figure, height: int = 340, **kwargs):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return st.plotly_chart(fig, width="stretch", **kwargs)


def usd(*columns) -> dict:
    return {c: USD for c in columns}


def network_key(network: do.Network) -> tuple:
    return tuple(int(pd.util.hash_pandas_object(getattr(network, t), index=False).to_numpy().sum()) for t in TABLES)


@st.cache_data(show_spinner="Generating the simulated network...")
def sample_network() -> do.Network:
    return do.generate_sample_network()


@st.cache_data(max_entries=24, show_spinner="Finding the cheapest allocation...")
def solve(key: tuple, _network: do.Network, _settings: do.Settings):
    return do.baseline_plan(_network, _settings), do.optimize(_network, _settings)


@st.cache_data(max_entries=24, show_spinner=False)
def why_infeasible(key: tuple, _network: do.Network, _settings: do.Settings) -> list:
    return do.diagnose(_network, _settings)


@st.cache_data(max_entries=24, show_spinner=False)
def options_table(key: tuple, _network: do.Network, _settings: do.Settings) -> pd.DataFrame:
    return do.lane_options(_network, _settings)


@st.cache_data(max_entries=24, show_spinner="Testing holding windows...")
def consolidation_windows(key: tuple, _network: do.Network, _plan: do.Result, _settings: do.Settings) -> pd.DataFrame:
    return do.consolidation_by_window(_network, _plan, _settings)


@st.cache_data(max_entries=12, show_spinner="Re-planning at each fuel surcharge...")
def fuel_table(key: tuple, _network: do.Network, _settings: do.Settings) -> pd.DataFrame:
    return do.fuel_sensitivity(_network, _settings)


@st.cache_data(max_entries=12, show_spinner="Re-planning at each growth rate...")
def growth_table(key: tuple, _network: do.Network, _settings: do.Settings) -> pd.DataFrame:
    return do.demand_growth(_network, _settings)


@st.cache_data(max_entries=12, show_spinner="Removing each DC in turn...")
def loss_table(key: tuple, _network: do.Network, _settings: do.Settings) -> pd.DataFrame:
    return do.dc_loss_table(_network, _settings)


@st.cache_data(max_entries=12, show_spinner="Choosing which DCs to operate...")
def open_close_tables(key: tuple, _network: do.Network, _settings: do.Settings):
    return do.open_close(_network, _settings), do.dc_count_table(_network, _settings)


def flow_map(result: do.Result, network: do.Network, colors: dict, only=None) -> go.Figure:
    """Customers coloured by the DC that serves them, with a line for every lane."""
    lanes = result.lanes if only is None else result.lanes[result.lanes["customer_id"].isin(only)]
    lanes = lanes.merge(network.customers[["customer_id", "latitude", "longitude"]], on="customer_id")
    lanes = lanes.merge(network.dcs[["dc", "latitude", "longitude"]].rename(columns={"latitude": "dc_lat", "longitude": "dc_lon"}), on="dc")
    lanes["size"] = np.sqrt(lanes["weight_lbs"].clip(lower=1.0))
    fig = px.scatter_geo(lanes, lat="latitude", lon="longitude", color="dc", size="size", size_max=13, scope="usa", hover_name="city",
                         hover_data={"dc": True, "mode": True, "miles": ":,.0f", "transit_days": True, "total_cost": ":,.0f", "size": False,
                                     "latitude": False, "longitude": False},
                         color_discrete_map=colors, category_orders={"dc": list(colors)})
    for dc, g in lanes.groupby("dc"):
        gap = np.full(len(g), np.nan)
        fig.add_trace(go.Scattergeo(lat=np.column_stack([g["dc_lat"], g["latitude"], gap]).ravel(),
                                    lon=np.column_stack([g["dc_lon"], g["longitude"], gap]).ravel(), mode="lines", opacity=0.45,
                                    line=dict(width=0.9, color=colors[dc]), hoverinfo="skip", showlegend=False))
    dcs = network.dcs
    fig.add_trace(go.Scattergeo(lat=dcs["latitude"], lon=dcs["longitude"], mode="markers+text", text=dcs["dc"], textposition="top center",
                                textfont=dict(color=DARK, size=11), hoverinfo="text", showlegend=False,
                                marker=dict(symbol="square", size=12, line=dict(width=1.5, color=DARK),
                                            color=[colors[d] if d in result.open_dcs else "#ffffff" for d in dcs["dc"]])))
    fig.update_geos(showland=True, landcolor="#f1f3f6", showlakes=False, subunitcolor="#c9d1dc")
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    return fig


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Distribution Optimizer")
st.sidebar.caption("Distribution cost optimization against today's allocation")
page = st.sidebar.radio("View", ["Savings overview", "Network map", "Assignment explorer", "Modes and consolidation", "Scenarios",
                                 "Data", "How it works"])

if "network" not in st.session_state:
    st.session_state.update(network=sample_network(), source="sample")
network: do.Network = st.session_state["network"]
if st.session_state["source"] != "yours":
    st.sidebar.info("Using a simulated US network. Load your own on the **Data** view.")

st.sidebar.subheader("Rules and assumptions")
use_rule = st.sidebar.checkbox("Apply a service rule", True)
max_days = st.sidebar.slider("Longest transit (days)", 1, 8, 4, disabled=not use_rule)
single = st.sidebar.checkbox("One DC per customer", False, help="Off: a customer may get a share of its shipments from each of several DCs")
choose = st.sidebar.checkbox("Let the plan close DCs", False, help="A closed DC ships nothing and pays no fixed cost")
fuel = st.sidebar.slider("Fuel surcharge (%)", 0, 60, 20, 5) / 100
modes = st.sidebar.multiselect("Modes allowed", list(do.MODES), default=list(do.MODES))
capacity_use = st.sidebar.slider("Usable DC capacity (%)", 60, 100, 100, 5) / 100
if not modes:
    st.sidebar.warning("Pick at least one mode; all three are used until you do.")
    modes = list(do.MODES)
settings = do.Settings(max_transit_days=max_days if use_rule else None, single_sourcing=single, choose_dcs=choose, modes=tuple(modes),
                       max_capacity_use=capacity_use, rates=do.RateCard(fuel_surcharge=fuel), time_limit=SOLVER_SECONDS)
key = (network_key(network), settings.max_transit_days, single, choose, settings.modes, capacity_use, fuel)
weeks = len(network.weeks)
st.sidebar.caption(f"{len(network.dcs)} DCs, {len(network.customers)} customers, {weeks} weeks of demand, "
                   f"{network.demand['units'].sum():,.0f} units.")

needs_plan = page not in ("Data", "How it works")
if needs_plan:
    baseline, plan = solve(key, network, settings)
    if not plan.ok:
        st.title("No allocation satisfies these rules")
        st.error(plan.message)
        fixes = why_infeasible(key, network, settings)
        if fixes:
            st.info("A plan becomes possible if you relax " + ", or ".join(f"**{f}**" for f in fixes) + ". Use the controls in the sidebar.")
        else:
            st.info("No single rule is to blame: more than one of the service rule, DC capacity, single sourcing and the allowed modes "
                    "would have to be relaxed together.")
        st.stop()
    comparison = do.compare(baseline, plan)
    dc_colors = {d: PALETTE[k % len(PALETTE)] for k, d in enumerate(network.dcs["dc"])}

# ========================================================== SAVINGS OVERVIEW
if page == "Savings overview":
    st.title("Estimated saving against today's allocation")
    c = comparison
    st.info(do.savings_headline(c, baseline, plan, settings))
    k = st.columns(5)
    k[0].metric("Estimated saving", money(c.saving), f"{c.saving_pct:.1%} of today's cost", delta_color="off")
    k[1].metric("Per week", money(c.saving_per_week))
    k[2].metric("Customers that change", f"{c.customers_changed} of {c.customers}")
    k[3].metric("Within the service rule", pct(plan.share_within_service, 1),
                f"{(plan.share_within_service - baseline.share_within_service) * 100:+.1f} pts" if use_rule else None)
    k[4].metric("Average transit days", f"{plan.average_transit_days:.2f}", f"{plan.average_transit_days - baseline.average_transit_days:+.2f}",
                delta_color="inverse")
    if plan.status != "optimal":
        st.warning(plan.message)

    left, right = st.columns(2)
    with left:
        parts = c.costs[c.costs["component"] != "Total"]
        fig = go.Figure()
        fig.add_bar(x=parts["component"], y=parts["baseline"], name="Today", marker_color=GREY)
        fig.add_bar(x=parts["component"], y=parts["optimized"], name="Recommended", marker_color=BLUE)
        fig.update_layout(barmode="group", yaxis_title=f"Cost over {weeks} weeks ($)", legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
    with right:
        fig = go.Figure(go.Waterfall(
            x=["Today", *parts["component"], "Recommended"], measure=["absolute", *["relative"] * len(parts), "total"],
            y=[baseline.total_cost, *parts["change"], plan.total_cost], decreasing=dict(marker=dict(color=GREEN)),
            increasing=dict(marker=dict(color=RED)), totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=GREY))))
        fig.update_layout(yaxis_title="Total cost ($)", showlegend=False)
        fig.update_yaxes(range=[min(baseline.total_cost, plan.total_cost) * 0.8, max(baseline.total_cost, plan.total_cost) * 1.03])
        show(fig, 340)
    st.caption("Left: each cost today and in the recommended plan. Right: how today's total becomes the plan's total. "
               "Both plans are costed with the same rates on the same demand.")
    if baseline.violations:
        st.warning("Rules that today's allocation breaks: " + "; ".join(baseline.violations) + ". The recommended plan obeys every rule, "
                   "which is part of why some customers move.")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Cost breakdown")
        st.dataframe(c.costs, hide_index=True, column_config={**usd("baseline", "optimized", "change"), "change_pct": PCT})
        by_type = c.lane_changes.groupby("change").agg(customers=("customer_id", "size"), saving=("saving", "sum")).reset_index()
        st.dataframe(by_type, hide_index=True, column_config=usd("saving"))
        st.caption("Saving by kind of change (freight and handling of the customers that change).")
    with right:
        st.subheader("Distribution centers: busiest week against capacity")
        load = baseline.dc_summary[["dc", "peak_week_units", "capacity_units_per_week"]].merge(
            plan.dc_summary[["dc", "peak_week_units", "open"]], on="dc", suffixes=("_today", "_plan"))
        fig = go.Figure()
        fig.add_bar(x=load["dc"], y=load["peak_week_units_today"], name="Today", marker_color=GREY)
        fig.add_bar(x=load["dc"], y=load["peak_week_units_plan"], name="Recommended", marker_color=BLUE)
        fig.add_scatter(x=load["dc"], y=load["capacity_units_per_week"] * capacity_use, mode="markers", name="Usable capacity",
                        marker=dict(symbol="line-ew-open", size=44, color=RED, line=dict(width=3)))
        fig.update_layout(barmode="group", yaxis_title="Units in the busiest week", legend=dict(orientation="h", y=-0.2))
        show(fig, 330)

    st.subheader("Take it with you")
    hold = do.consolidation_analysis(network, plan, settings)
    extra = {} if single else {"dc_loss": loss_table(key, network, settings), "fuel": fuel_table(key, network, settings)}
    report = do.build_report(baseline, plan, c, settings, consolidation=hold, simulated=st.session_state["source"] != "yours", **extra)
    d = st.columns(3)
    d[0].download_button("Optimization report (HTML)", report, "optimization_report.html", "text/html", width="stretch", type="primary")
    d[1].download_button("Recommended assignment (CSV)", plan.lanes.to_csv(index=False), "recommended_assignment.csv", "text/csv", width="stretch")
    d[2].download_button("Lanes that change (CSV)", c.lane_changes.to_csv(index=False), "lane_changes.csv", "text/csv", width="stretch")
    st.caption("All figures are estimates from the rate functions and demand in use, not realized savings. The rate functions are "
               "illustrative, not real carrier tariffs."
               + (" With one DC per customer, the report leaves out the DC-loss and fuel sections to keep this page quick." if single else ""))

# =============================================================== NETWORK MAP
elif page == "Network map":
    st.title("Who serves whom: today and recommended")
    c = comparison
    miles = c.service.set_index("measure").loc["Average road miles per unit"]
    closed = [d for d in network.dcs["dc"] if d not in plan.open_dcs]
    st.info(f"{c.dc_changes} of {c.customers} customers move to a different DC. The average unit travels {miles['optimized']:,.0f} road miles "
            f"in the recommended plan against {miles['baseline']:,.0f} today."
            + (f" The plan closes {', '.join(closed)} (white square)." if closed else ""))
    k = st.columns(4)
    k[0].metric("Customers that move DC", c.dc_changes)
    k[1].metric("Road miles per unit", f"{miles['optimized']:,.0f}", f"{miles['optimized'] - miles['baseline']:+,.0f}", delta_color="inverse")
    k[2].metric("DCs open", len(plan.open_dcs), len(plan.open_dcs) - len(baseline.open_dcs) or None)
    k[3].metric("Customers split between DCs", plan.split_customers)
    only_moved = st.checkbox("Show only customers whose DC changes")
    moved = set(c.lane_changes.loc[c.lane_changes["change"] != "mode", "customer_id"]) if only_moved else None
    left, right = st.columns(2)
    with left:
        st.subheader("Today")
        show(flow_map(baseline, network, dc_colors, moved), 430)
    with right:
        st.subheader("Recommended")
        show(flow_map(plan, network, dc_colors, moved), 430)
    st.caption("Squares are distribution centers; each dot is a customer, sized by weight shipped and coloured by the DC that serves it. "
               "Lines are straight for display only; distances are estimated road miles.")
    by_region = (baseline.lanes.groupby(["region", "dc"])["units"].sum().rename("today").to_frame()
                 .join(plan.lanes.groupby(["region", "dc"])["units"].sum().rename("recommended"), how="outer").fillna(0.0).reset_index())
    long = by_region.melt(id_vars=["region", "dc"], var_name="plan", value_name="units")
    fig = px.bar(long, x="plan", y="units", color="dc", facet_col="region", color_discrete_map=dc_colors,
                 category_orders={"dc": list(dc_colors), "plan": ["today", "recommended"]}, labels={"units": "Units", "plan": ""})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    show(fig, 320)
    st.caption("Units delivered to each Census region, by the DC they ship from.")

# ======================================================= ASSIGNMENT EXPLORER
elif page == "Assignment explorer":
    st.title("The recommended assignment, customer by customer")
    c = comparison
    top = c.lane_changes.iloc[0] if len(c.lane_changes) else None
    st.info(f"{c.customers_changed} of {c.customers} customers change: {c.dc_changes} move to another DC and {c.mode_changes} change mode."
            + (f" The largest single change is {top['customer_id']} ({top['city']}, {top['state']}): {top['from_dc']} by {top['from_mode']} to "
               f"{top['to_dc']} by {top['to_mode']}, an estimated {money(top['saving'])} over {weeks} weeks." if top is not None else ""))
    today = baseline.lanes.groupby("customer_id").agg(dc_today=("dc", " + ".join), mode_today=("mode", " + ".join), cost_today=("total_cost", "sum"))
    table = plan.lanes.merge(today, on="customer_id")
    table["customer_cost"] = table.groupby("customer_id")["total_cost"].transform("sum")
    table["saving"] = (table["cost_today"] - table["customer_cost"]) * table["share"]
    table["changes"] = table["customer_id"].isin(set(c.lane_changes["customer_id"]))
    k = st.columns(4)
    k[0].metric("Lanes in the plan", len(table))
    k[1].metric("Customers that change", c.customers_changed)
    k[2].metric("Largest saving on one customer", money(c.lane_changes["saving"].max() if len(c.lane_changes) else 0))
    k[3].metric("Customers that cost more", int((c.lane_changes["saving"] < 0).sum()),
                help="They give up DC capacity to others or move to meet the service rule; the total is what is minimized")

    f = st.columns(4)
    pick_dc = f[0].multiselect("DC", list(network.dcs["dc"]))
    pick_region = f[1].multiselect("Region", sorted(network.customers["region"].unique()))
    pick_mode = f[2].multiselect("Mode", list(settings.modes))
    only_changed = f[3].checkbox("Only customers that change", False)
    view = table
    if pick_dc:
        view = view[view["dc"].isin(pick_dc)]
    if pick_region:
        view = view[view["region"].isin(pick_region)]
    if pick_mode:
        view = view[view["mode"].isin(pick_mode)]
    if only_changed:
        view = view[view["changes"]]
    cols = ["customer_id", "city", "state", "region", "dc_today", "dc", "share", "mode_today", "mode", "miles", "transit_days", "units",
            "weight_lbs", "freight_cost", "handling_cost", "total_cost", "saving"]
    st.dataframe(view[cols].sort_values("saving", ascending=False), hide_index=True, height=330, column_config={
        "dc": "dc_recommended", "mode": "mode_recommended", "share": PCT, "miles": NUM, "units": NUM, "weight_lbs": NUM,
        **usd("freight_cost", "handling_cost", "total_cost", "saving")})
    st.caption(f"{len(view)} lane(s). Costs are freight plus handling over {weeks} weeks; saving is against how the customer is served today.")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(c.lane_changes, x="cost_before", y="cost_after", color="change", hover_name="city",
                         hover_data={"from_dc": True, "to_dc": True, "from_mode": True, "to_mode": True, "saving": ":,.0f"},
                         color_discrete_map={"DC": BLUE, "mode": AMBER, "DC and mode": GREEN},
                         labels={"cost_before": "Cost today ($)", "cost_after": "Cost in the plan ($)"}, log_x=True, log_y=True)
        if len(c.lane_changes):
            span = [float(c.lane_changes[["cost_before", "cost_after"]].min().min()), float(c.lane_changes[["cost_before", "cost_after"]].max().max())]
            fig.add_scatter(x=span, y=span, mode="lines", line=dict(dash="dash", color=GREY), showlegend=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        show(fig, 340)
        st.caption("Each point is a customer that changes. Below the dashed line it costs less in the plan than today.")
    with right:
        ids = list(network.customers["customer_id"])
        labels = dict(zip(ids, network.customers["customer_id"] + "  " + network.customers["city"] + ", " + network.customers["state"]))
        default = ids.index(top["customer_id"]) if top is not None else 0
        chosen = st.selectbox("Compare the DCs for one customer", ids, index=default, format_func=labels.get)
        opts = options_table(key, network, settings)
        opts = opts[opts["customer_id"] == chosen]
        best = opts[opts["within_service"]].sort_values("total_cost").groupby("dc", as_index=False).first()
        served_today = set(baseline.lanes.loc[baseline.lanes["customer_id"] == chosen, "dc"])
        served_plan = set(plan.lanes.loc[plan.lanes["customer_id"] == chosen, "dc"])
        best["status"] = [("recommended" if d in served_plan else "today" if d in served_today else "not open in the plan"
                           if d not in plan.open_dcs else "other") for d in best["dc"]]
        fig = px.bar(best.sort_values("total_cost"), x="total_cost", y="dc", orientation="h", color="status", text="mode",
                     color_discrete_map={"recommended": BLUE, "today": AMBER, "other": GREY, "not open in the plan": "#d9dee5"},
                     labels={"total_cost": f"Freight + handling over {weeks} weeks ($), cheapest mode within the rule", "dc": ""})
        fig.update_yaxes(categoryorder="total descending")
        fig.update_layout(legend=dict(orientation="h", y=-0.3))
        show(fig, 290)
        st.caption("DCs that cannot reach the customer within the service rule by any allowed mode are not shown. The cheapest DC is "
                   "not always the recommended one: capacity has to be shared.")
    st.download_button("Recommended assignment (CSV)", plan.lanes.to_csv(index=False), "recommended_assignment.csv", "text/csv")

# ==================================================== MODES AND CONSOLIDATION
elif page == "Modes and consolidation":
    st.title("Modes, and whether to ship less often")
    max_hold = st.sidebar.slider("Longest hold for consolidation (working days)", 0, 4, 2)
    hold = do.consolidation_analysis(network, plan, settings, max_hold)
    s = do.consolidation_summary(hold)
    modes_only = do.baseline_with_best_modes(network, settings)
    mode_saving = baseline.total_cost - modes_only.total_cost
    st.info(f"Picking the cheapest mode on today's lanes, with nobody changing DC, would save an estimated {money(mode_saving)} over {weeks} weeks; "
            f"the full plan changes the mode of {comparison.mode_changes} customers. On top of the plan, holding orders up to {max_hold} working "
            f"day(s) would consolidate {s['lanes_consolidated']} of {s['lanes']} lanes and cut freight by a further estimated {money(s['saving'])} "
            f"({s['saving_pct']:.1%}), before any cost of holding goods.")
    k = st.columns(4)
    k[0].metric("Saving from mode choice alone", money(mode_saving))
    k[1].metric("Customers that change mode", comparison.mode_changes)
    k[2].metric("Consolidation saving", money(s["saving"]), f"{s['saving_pct']:.1%} of plan freight", delta_color="off")
    k[3].metric("Scheduled shipments a week", s["shipments_per_week_after"], s["shipments_per_week_after"] - s["shipments_per_week_before"] or None,
                delta_color="inverse")

    st.subheader("Mode mix")
    mix = pd.concat([baseline.mode_mix.assign(plan="Today"), plan.mode_mix.assign(plan="Recommended")])
    left, right = st.columns(2)
    with left:
        fig = px.bar(mix, x="plan", y="freight_cost", color="mode", color_discrete_map=MODE_COLORS, category_orders={"mode": list(do.MODES)},
                     labels={"freight_cost": f"Freight over {weeks} weeks ($)", "plan": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 320)
    with right:
        fig = px.bar(mix, x="plan", y="lanes", color="mode", color_discrete_map=MODE_COLORS, category_orders={"mode": list(do.MODES)},
                     labels={"lanes": "Lanes", "plan": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        show(fig, 320)
    st.dataframe(mix[["plan", "mode", "lanes", "shipments", "weight_lbs", "freight_cost"]], hide_index=True,
                 column_config={"shipments": NUM, "weight_lbs": NUM, **usd("freight_cost")})
    st.caption("Shipments are packages, LTL shipments or trucks. A lane uses the cheapest allowed mode that delivers within the service rule.")

    st.subheader("What one shipment costs by mode")
    miles = st.slider("Distance (road miles)", 50, 2500, 600, 50)
    weights = np.unique(np.round(np.logspace(0, np.log10(45000), 160)))
    curve = pd.DataFrame(do.rate_curve(miles, weights, settings.rates)).assign(weight_lbs=weights).melt("weight_lbs", var_name="mode", value_name="cost")
    curve = curve[curve["mode"].isin(settings.modes)]
    fig = px.line(curve, x="weight_lbs", y="cost", color="mode", log_x=True, log_y=True, color_discrete_map=MODE_COLORS,
                  labels={"weight_lbs": "Shipment weight (lb)", "cost": "Charge with fuel surcharge ($)"})
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    show(fig, 330)
    days = ", ".join(f"{m} {int(do.transit_days(m, miles, settings.rates))} day(s)" for m in settings.modes)
    st.caption(f"Transit at {miles:,} miles: {days}. These rate functions are illustrative (a minimum charge, weight breaks, cost rising "
               "with distance, a fuel surcharge) and are not real carrier tariffs. Parcel steps up with every extra 150 lb package.")

    st.subheader("Consolidation: fewer, larger shipments")
    left, right = st.columns([2, 3])
    with left:
        windows = consolidation_windows(key, network, plan, settings)
        fig = px.bar(windows, x="max_hold_days", y="saving", color_discrete_sequence=[GREEN], text_auto=",.0f",
                     labels={"max_hold_days": "Longest hold (working days)", "saving": f"Freight saving over {weeks} weeks ($)"})
        show(fig, 320)
        st.caption("Holding orders up to h days means shipping at most once every h + 1 working days. A hold is used only where days "
                   "held plus transit days stay inside the service rule.")
    with right:
        held = hold[hold["hold_days"] > 0]
        st.dataframe(held[["customer_id", "city", "dc", "shipments_per_week", "consolidated_shipments_per_week", "hold_days", "mode",
                           "consolidated_mode", "order_to_delivery_days", "consolidated_lbs_per_shipment", "freight_before", "freight_after",
                           "saving"]], hide_index=True, height=320,
                     column_config={"consolidated_lbs_per_shipment": NUM, **usd("freight_before", "freight_after", "saving")})
    st.download_button("Consolidation analysis (CSV)", hold.to_csv(index=False), "consolidation.csv", "text/csv")
    st.caption("Only freight is counted. Inventory held, dock space and the customer's view of fewer deliveries are not.")

# ================================================================= SCENARIOS
elif page == "Scenarios":
    st.title("What if conditions change?")
    kind = st.radio("Scenario", ["Fuel surcharge", "Demand growth", "Loss of a DC", "Which DCs to operate"], horizontal=True)
    if single:
        st.caption(f"With one DC per customer each re-plan may take up to {SOLVER_SECONDS:.0f} seconds when capacity is nearly full.")
    if kind == "Fuel surcharge":
        t = fuel_table(key, network, settings)
        per_ten = (t["baseline_cost"].iloc[-1] - t["baseline_cost"].iloc[0]) / (t["fuel_surcharge"].iloc[-1] - t["fuel_surcharge"].iloc[0]) / 10
        per_ten_plan = (t["keep_plan_cost"].iloc[-1] - t["keep_plan_cost"].iloc[0]) / (t["fuel_surcharge"].iloc[-1] - t["fuel_surcharge"].iloc[0]) / 10
        st.info(f"Every 10 points of fuel surcharge adds about {money(per_ten)} to today's allocation and {money(per_ten_plan)} to the "
                f"recommended plan over {weeks} weeks, so the plan's saving grows with fuel. Re-planning for a different fuel price adds at most "
                f"{money(t['reoptimizing_gain'].max())}: the surcharge raises every lane's freight by the same percentage, so the best "
                "assignment barely moves.")
        k = st.columns(3)
        k[0].metric("Cost of 10 fuel points, today", money(per_ten))
        k[1].metric("Cost of 10 fuel points, plan", money(per_ten_plan))
        k[2].metric("Most that re-planning adds", money(t["reoptimizing_gain"].max()))
        fig = go.Figure()
        fig.add_scatter(x=t["fuel_surcharge"], y=t["baseline_cost"], name="Today's allocation", mode="lines+markers", line_color=GREY)
        fig.add_scatter(x=t["fuel_surcharge"], y=t["keep_plan_cost"], name="Recommended plan, unchanged", mode="lines+markers", line_color=BLUE)
        fig.add_scatter(x=t["fuel_surcharge"], y=t["reoptimized_cost"], name="Re-planned at that surcharge", mode="lines+markers",
                        line=dict(color=GREEN, dash="dot"))
        fig.add_vline(x=fuel, line_dash="dash", line_color=RED)
        fig.update_layout(xaxis_title="Fuel surcharge", yaxis_title=f"Total cost over {weeks} weeks ($)", legend=dict(orientation="h", y=-0.25))
        fig.update_xaxes(tickformat=".0%")
        show(fig, 360)
        st.dataframe(t, hide_index=True, column_config={"fuel_surcharge": PCT, **usd("baseline_cost", "keep_plan_cost", "reoptimized_cost",
                                                                                     "saving_vs_baseline", "reoptimizing_gain")})
        st.download_button("Fuel table (CSV)", t.to_csv(index=False), "scenario_fuel.csv", "text/csv")
    elif kind == "Demand growth":
        t = growth_table(key, network, settings)
        ok, bad = t[t["status"] != "infeasible"], t[t["status"] == "infeasible"]
        text = (f"The network can serve demand up to {ok['growth'].max():+.0%} within the rules." if len(ok)
                else "No tested growth rate has a plan within the rules.")
        if len(bad):
            text += f" At {bad['growth'].min():+.0%} no plan exists: {bad['note'].iloc[0]}"
        if len(ok) > 1:
            text += (f" Cost per unit goes from {ok['cost_per_unit'].iloc[0]:.2f} to {ok['cost_per_unit'].iloc[-1]:.2f} dollars across the "
                     "feasible range, as fixed cost is spread over more units.")
        st.info(text)
        k = st.columns(3)
        k[0].metric("Largest growth with a plan", f"{ok['growth'].max():+.0%}" if len(ok) else "none")
        k[1].metric("DCs full at that growth", int(ok["dcs_full"].iloc[-1]) if len(ok) else 0)
        k[2].metric("Customers split at that growth", int(ok["split_customers"].iloc[-1]) if len(ok) else 0)
        fig = go.Figure()
        for name, color in (("freight_cost", BLUE), ("handling_cost", AMBER), ("fixed_cost", GREY)):
            fig.add_bar(x=ok["growth"], y=ok[name], name=name.replace("_", " "), marker_color=color)
        fig.add_scatter(x=ok["growth"], y=ok["cost_per_unit"], name="cost per unit (right axis)", yaxis="y2", mode="lines+markers", line_color=RED)
        fig.update_layout(barmode="stack", xaxis_title="Change in demand", yaxis_title=f"Total cost over {weeks} weeks ($)",
                          yaxis2=dict(overlaying="y", side="right", title="$ per unit", showgrid=False), legend=dict(orientation="h", y=-0.25))
        fig.update_xaxes(tickformat="+.0%")
        show(fig, 360)
        st.dataframe(t, hide_index=True, column_config={"growth": PCT, "units": NUM, "peak_utilization": PCT, "cost_per_unit": st.column_config.NumberColumn(format="$%.3f"),
                                                        **usd("total_cost", "freight_cost", "handling_cost", "fixed_cost")})
        st.caption("Every demand line is scaled by the same rate. Peak utilization is the fullest DC in its busiest week.")
        st.download_button("Growth table (CSV)", t.to_csv(index=False), "scenario_growth.csv", "text/csv")
    elif kind == "Loss of a DC":
        t = loss_table(key, network, settings)
        absorbed, stuck = t[t["outcome"] == "absorbed"], t[t["outcome"] != "absorbed"]
        text = ""
        if len(stuck):
            text += (f"Losing {', '.join(stuck['dc_lost'])} leaves no plan within the rules "
                     f"(would need: {'; '.join(sorted(set(r for r in stuck['relax'] if r))) or 'more than one rule relaxed'}). ")
        if len(absorbed):
            worst, least = absorbed.iloc[0], absorbed.iloc[-1]
            text += (f"Of the losses the network can absorb, {worst['dc_lost']} is the most expensive: an estimated {money(worst['extra_cost'])} more "
                     f"over {weeks} weeks ({worst['extra_cost_pct']:+.0%}); {least['dc_lost']} is the cheapest at {money(least['extra_cost'])}.")
        st.info(text or "No DC can be lost.")
        k = st.columns(3)
        k[0].metric("DCs whose loss can be absorbed", f"{len(absorbed)} of {len(t)}")
        k[1].metric("Most expensive absorbed loss", money(absorbed["extra_cost"].max()) if len(absorbed) else "n/a")
        k[2].metric("Cheapest loss", money(absorbed["extra_cost"].min()) if len(absorbed) else "n/a")
        fig = go.Figure(go.Bar(x=absorbed["extra_cost"], y=absorbed["dc_lost"], orientation="h", marker_color=RED))
        fig.update_layout(xaxis_title=f"Extra cost over {weeks} weeks after re-planning ($)")
        fig.update_yaxes(autorange="reversed")
        show(fig, 300)
        st.dataframe(t, hide_index=True, column_config={"units_affected": NUM, "extra_cost_pct": PCT, "share_within_service": PCT,
                                                        "average_transit_days": st.column_config.NumberColumn(format="%.2f"), **usd("extra_cost")})
        st.caption("Each DC is removed in turn and the network re-planned under the same rules. Extra cost leaves out the lost DC's own "
                   "fixed cost, because whether it is still paid depends on why the DC is lost. A very large extra cost usually means the "
                   "service rule can then only be met with a much dearer mode; relax the rule in the sidebar to see.")
        st.download_button("DC loss table (CSV)", t.to_csv(index=False), "scenario_dc_loss.csv", "text/csv")
    else:
        out, t = open_close_tables(key, network, settings)
        chosen = out["chosen"]
        if not chosen.ok:
            st.error(chosen.message)
            st.stop()
        closed = out["closed"]
        st.info((f"Free to choose, the plan keeps {len(chosen.open_dcs)} of {len(network.dcs)} DCs and closes {', '.join(closed)}: the fixed cost "
                 f"saved is larger than the freight and handling added, for an estimated {money(out['saving'])} over {weeks} weeks against "
                 "keeping every DC open." if closed else "Keeping every DC open is the cheapest network: no DC's fixed cost is larger than "
                 "the freight and handling it saves.") + " Closing costs, lease terms and the risk of running fewer sites are not counted.")
        k = st.columns(3)
        k[0].metric("DCs in the cheapest network", f"{len(chosen.open_dcs)} of {len(network.dcs)}")
        k[1].metric("Saving against all open", money(out["saving"]))
        k[2].metric("Average transit days", f"{chosen.average_transit_days:.2f}", f"{chosen.average_transit_days - out['all_open'].average_transit_days:+.2f}",
                    delta_color="inverse")
        ok = t[t["status"] != "infeasible"]
        fig = go.Figure()
        for name, color in (("freight_cost", BLUE), ("handling_cost", AMBER), ("fixed_cost", GREY)):
            fig.add_bar(x=ok["dcs_open"], y=ok[name], name=name.replace("_", " "), marker_color=color)
        fig.update_layout(barmode="stack", xaxis_title="Number of DCs open (best choice for each number)", yaxis_title=f"Total cost over {weeks} weeks ($)",
                          legend=dict(orientation="h", y=-0.25), xaxis=dict(dtick=1))
        show(fig, 340)
        st.dataframe(t, hide_index=True, column_config={"share_within_service": PCT, "average_transit_days": st.column_config.NumberColumn(format="%.2f"),
                                                        **usd("total_cost", "freight_cost", "handling_cost", "fixed_cost")})
        st.caption("For each number of DCs the optimizer picks which to keep. Numbers with no row of costs cannot serve all demand within the rules.")
        st.download_button("DC count table (CSV)", t.to_csv(index=False), "scenario_open_close.csv", "text/csv")

# ====================================================================== DATA
elif page == "Data":
    st.title("Data")
    st.info("A network is five tables: DCs, customers, product families, weekly demand and today's allocation. "
            + ("The network in use is simulated; none of it describes a real company." if st.session_state["source"] != "yours"
               else "The network in use is the one you loaded."))
    k = st.columns(5)
    units, weight = network.weekly()
    k[0].metric("DCs", len(network.dcs))
    k[1].metric("Customers", len(network.customers))
    k[2].metric("Weeks of demand", weeks)
    k[3].metric("Units a week", f"{units.sum() / max(weeks, 1):,.0f}")
    k[4].metric("Busiest week against capacity", pct(units.sum(axis=0).max() / network.dcs["capacity_units_per_week"].sum()))

    st.subheader("Edit the distribution centers")
    st.caption("Change capacity, handling cost or fixed cost, then apply. Every view is re-planned with the new values.")
    editable = ["capacity_units_per_week", "handling_cost_per_unit", "fixed_cost_per_week"]
    edited = st.data_editor(network.dcs.drop(columns="region"), hide_index=True, key="dc_editor",
                            disabled=[c for c in network.dcs.columns if c not in editable],
                            column_config={"capacity_units_per_week": st.column_config.NumberColumn(min_value=0, step=500),
                                           "handling_cost_per_unit": st.column_config.NumberColumn(min_value=0.0, format="$%.2f"),
                                           "fixed_cost_per_week": st.column_config.NumberColumn(min_value=0.0, format="dollar")})
    c = st.columns(3)
    if c[0].button("Apply DC changes", type="primary"):
        try:
            changed = do.validate_network(edited, network.customers, network.products, network.demand, network.baseline)
        except do.SchemaError as exc:
            st.error(f"Could not apply: {exc}")
        else:
            st.session_state.update(network=changed, source=st.session_state["source"] if st.session_state["source"] == "yours" else "edited sample")
            st.rerun()
    if c[1].button("Reset to simulated data"):
        st.session_state.update(network=sample_network(), source="sample")
        st.session_state.pop("dc_editor", None)
        st.rerun()
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in TABLES:
            z.writestr(f"{name}.csv", getattr(network, name).drop(columns=["region"], errors="ignore").to_csv(index=False))
    c[2].download_button("Download current data (zip of CSV files)", archive.getvalue(), "network.zip", "application/zip")

    st.subheader("Load your own network")
    st.caption("Upload any of dcs.csv, customers.csv, products.csv, demand.csv and baseline.csv (the file names decide which table each is). "
               "Tables you do not upload are kept from the network in use.")
    ups = st.file_uploader("Network CSV files", type="csv", accept_multiple_files=True)
    if ups and st.button("Use these files"):
        given = {u.name.rsplit(".", 1)[0].lower(): u for u in ups}
        unknown = sorted(set(given) - set(TABLES))
        try:
            if unknown:
                raise do.SchemaError(f"file name(s) not recognised: {', '.join(unknown)}; expected {', '.join(t + '.csv' for t in TABLES)}")
            tables = [pd.read_csv(BytesIO(given[t].getvalue()), dtype={"dc": str, "customer_id": str, "week": str}) if t in given
                      else getattr(network, t) for t in TABLES]
            loaded = do.validate_network(*tables)
        except (do.SchemaError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not use these files: {exc}")
        else:
            st.session_state.update(network=loaded, source="yours")
            st.session_state.pop("dc_editor", None)
            st.rerun()

    st.subheader("Columns")
    st.dataframe(pd.DataFrame(do.COLUMN_HELP, columns=["table", "column", "required", "meaning"]), hide_index=True, height=320)

    st.subheader("Current data")
    left, right = st.columns(2)
    with left:
        weekly = pd.DataFrame({"week": network.weeks, "units": units.sum(axis=0), "pounds": weight.sum(axis=0)})
        fig = px.bar(weekly, x="week", y="units", color_discrete_sequence=[BLUE], labels={"week": "", "units": "Units ordered"})
        fig.add_hline(y=float(network.dcs["capacity_units_per_week"].sum()), line_dash="dash", line_color=RED,
                      annotation_text="capacity of all DCs")
        show(fig, 300)
    with right:
        per = pd.DataFrame({"pounds_per_shipment": weight.mean(axis=1) / network.customers["shipments_per_week"].to_numpy()})
        fig = px.histogram(per, x=np.log10(per["pounds_per_shipment"].clip(lower=1)), nbins=30, color_discrete_sequence=[GREEN],
                           labels={"x": "Typical pounds per shipment (log scale: 2 = 100 lb, 4 = 10,000 lb)"})
        fig.update_layout(yaxis_title="Customers")
        show(fig, 300)
    tabs = st.tabs(list(TABLES))
    for tab, name in zip(tabs, TABLES):
        tab.dataframe(getattr(network, name).head(500), hide_index=True, height=280)

# ============================================================== HOW IT WORKS
else:
    st.title("How it works")
    st.info("The optimizer costs today's allocation and the cheapest allocation that obeys your rules with the same rate functions on the "
            "same demand, and reports the difference as an estimated saving.")
    r = settings.rates
    k = st.columns(4)
    k[0].metric("Fuel surcharge", f"{r.fuel_surcharge:.0%}")
    k[1].metric("Parcel limit", f"{r.parcel_max_lbs:,.0f} lb")
    k[2].metric("LTL minimum charge", f"${r.ltl_min:,.0f}")
    k[3].metric("Truckload per mile", f"${r.tl_per_mile:.2f}")
    weights = np.unique(np.round(np.logspace(0, np.log10(45000), 120)))
    curve = pd.DataFrame(do.rate_curve(600, weights, r)).assign(weight_lbs=weights).melt("weight_lbs", var_name="mode", value_name="cost")
    curve["per_lb"] = curve["cost"] / curve["weight_lbs"]
    fig = px.line(curve, x="weight_lbs", y="per_lb", color="mode", log_x=True, log_y=True, color_discrete_map=MODE_COLORS,
                  labels={"weight_lbs": "Shipment weight (lb)", "per_lb": "Charge per pound at 600 miles ($)"})
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    show(fig, 300)
    st.caption("Why mode and shipment size matter: the charge per pound falls steeply as a shipment gets heavier, and each mode is "
               "cheapest over a different range of weights. Illustrative rate functions, not real tariffs.")
    st.markdown(
        f"""
**What is costed**

- *Freight.* A customer receives a set number of shipments each week; each carries that week's pounds divided equally. A shipment's
  charge depends on mode, road miles and weight, with a minimum charge and a fuel surcharge on every mode. Parcel is charged per
  package up to {r.parcel_max_lbs:,.0f} lb; LTL per 100 lb with cheaper rates at heavier weight breaks; truckload per truck per mile.
- *Handling.* Each DC's cost per unit shipped.
- *Fixed cost.* Each open DC's weekly fixed cost, whatever it ships.

Road miles are straight-line miles times 1.18. Transit days come from the distance and the mode's miles per day, plus one handling
day for LTL.

**What the optimizer decides**

It picks the DC (or DCs) for every customer to minimize freight plus handling, and fixed cost when it may close DCs, subject to:
all demand served; each DC within its capacity in every week; every lane delivered within the service rule by the cheapest allowed
mode that can do it; and, if you ask, one DC per customer. It is solved exactly as a linear or mixed-integer program
(scipy `milp`, HiGHS). When no plan exists, the dashboard names the rule to relax.

**Splitting a customer** means a share of its shipments comes from each DC, each shipment at its usual size, so costs and DC load
scale with the share. Shares are not rounded to whole shipments.

**Comparison with today.** Today's allocation is costed exactly as recorded, with its own DCs and modes, by the same functions.
Rules it breaks are listed rather than hidden. The difference is an estimate, not a realized saving.

**Consolidation** asks a separate question: on each lane of the plan, would holding orders a few days and shipping less often be
cheaper, with days held plus transit days still inside the service rule?

**Scenarios** re-plan under a different fuel surcharge, more or less demand, the loss of each DC, and a free choice of which DCs
to operate.

**Limits.** The sample network is simulated. The rate functions are illustrative and are not carrier tariffs: replace them with
contract rates before acting. Demand is taken as given, week by week, with no forecast or uncertainty. Shipments are the same size
within a week. The model does not count inventory, the cost of moving a customer to another DC, closing costs, carrier capacity,
or delivery appointments. Distances are estimates, not routed miles.
"""
    )
