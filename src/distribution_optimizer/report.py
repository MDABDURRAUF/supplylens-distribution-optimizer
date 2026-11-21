"""Printable output: the optimization report as a single HTML file, and the plain-language summary."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .consolidation import consolidation_summary
from .optimizer import Comparison, Result, Settings


def money(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def pct(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def _num(x, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2430}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 8px;font-size:19px} p,li{line-height:1.5}
p.note{color:#56627a;font-size:14px;margin:0 0 12px}
.lead{font-size:16px;background:#fff;border:1px solid #dfe3ea;border-left:5px solid #3a6ea5;border-radius:8px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.kpi{background:#fff;border:1px solid #dfe3ea;border-radius:8px;padding:14px}
.kpi b{display:block;font-size:21px} .kpi span{font-size:13px;color:#56627a}
.scroll{overflow-x:auto;background:#fff;border:1px solid #dfe3ea;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #edf0f4;vertical-align:top}
th{background:#eef1f6;font-weight:600;white-space:nowrap} td.num{text-align:right;white-space:nowrap}
.caveat{background:#fff8e6;border:1px solid #f0d58a;border-radius:8px;padding:12px 14px;font-size:14px}
@media print{body{background:#fff}}
"""


def _table(df: pd.DataFrame, formats: dict | None = None, empty: str = "Nothing to show.") -> str:
    if df is None or df.empty:
        return f'<p class="note">{escape(empty)}</p>'
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c).replace('_', ' '))}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            if c in formats:
                cells.append(f'<td class="num">{escape(formats[c](v))}</td>')
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in items) + "</div>"


def _page(title: str, subtitle: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main><h1>{escape(title)}</h1>"
            f'<p class="note">{escape(subtitle)}</p>{body}</main></body></html>')


def rules_text(settings: Settings) -> str:
    """The rules and assumptions of a plan in one sentence."""
    parts = ["no service rule" if settings.max_transit_days is None else f"delivery within {settings.max_transit_days} transit days",
             "one DC per customer" if settings.single_sourcing else "customers may be split between DCs",
             "the plan may close DCs" if settings.choose_dcs else "every DC stays open",
             "DC capacity ignored" if settings.max_capacity_use is None else f"DCs used up to {settings.max_capacity_use:.0%} of weekly capacity",
             f"modes: {', '.join(settings.modes)}", f"fuel surcharge {settings.rates.fuel_surcharge:.0%}"]
    return "; ".join(parts) + "."


def savings_headline(comparison: Comparison, baseline: Result, optimized: Result, settings: Settings = Settings()) -> str:
    """One paragraph: the estimated saving, what changes, and what happens to service."""
    c = comparison
    weeks = f"{c.weeks} week{'s' if c.weeks != 1 else ''}"
    if c.saving > 0:
        text = (f"Estimated saving: {money(c.saving)} over {weeks} of demand ({pct(c.saving_pct, 1)} of today's {money(baseline.total_cost)}), "
                f"about {money(c.saving_per_week)} a week. ")
    else:
        text = (f"No cheaper allocation was found under these rules: the recommended plan costs {money(-c.saving)} more than today's "
                f"{money(baseline.total_cost)} over {weeks}, which is the price of rules that today's allocation breaks. ")
    text += (f"{c.customers_changed} of {c.customers} customers change: {c.dc_changes} move to another DC and {c.mode_changes} change mode. ")
    closed = [d for d in baseline.open_dcs if d not in optimized.open_dcs]
    if closed:
        text += f"The plan closes {', '.join(closed)}. "
    if settings.max_transit_days is not None:
        text += (f"Units delivered within the {settings.max_transit_days}-day rule go from {pct(baseline.share_within_service, 1)} to "
                 f"{pct(optimized.share_within_service, 1)}; ")
    text += f"average transit goes from {baseline.average_transit_days:.2f} to {optimized.average_transit_days:.2f} days. "
    text += "This is an estimate from the rates and demand supplied, not a realized saving."
    return text


def build_report(baseline: Result, optimized: Result, comparison: Comparison, settings: Settings = Settings(),
                 consolidation: pd.DataFrame | None = None, dc_loss: pd.DataFrame | None = None, fuel: pd.DataFrame | None = None,
                 simulated: bool = False, top: int = 40) -> str:
    """The optimization report as one HTML page.

    Summary, cost breakdown against today, DC loads, the lanes that change,
    mode mix and service; plus, when the tables are passed in, consolidation
    (``consolidation_analysis``), loss of a DC (``dc_loss_table``) and fuel
    (``fuel_sensitivity``). ``simulated=True`` adds a notice that the data is made up.
    """
    c = comparison
    body = f'<p class="lead">{escape(savings_headline(c, baseline, optimized, settings))}</p>'
    if simulated:
        body += ('<p class="caveat">The network, demand and today\'s allocation in this report are simulated. The figures show how the '
                 "method works; they are not findings about any real company.</p>")
    body += _kpis([("estimated saving", money(c.saving)), ("of today's cost", pct(c.saving_pct, 1)), ("per week", money(c.saving_per_week)),
                   ("customers that change", f"{c.customers_changed} of {c.customers}"),
                   ("within service rule", f"{pct(baseline.share_within_service, 1)} to {pct(optimized.share_within_service, 1)}")])
    body += f'<p class="note">Rules and assumptions: {escape(rules_text(settings))}</p>'
    if optimized.status != "optimal":
        body += f'<p class="note">{escape(optimized.message)}</p>'
    if optimized.split_customers:
        body += (f'<p class="note">{optimized.split_customers} customer(s) are split between DCs to fit capacity: a share of their '
                 "shipments comes from each DC, shown as a percentage in the tables below.</p>")

    body += f"<h2>Cost against today</h2><p class=\"note\">Both plans are costed with the same rates on the same {c.weeks} weeks of demand.</p>"
    body += _table(c.costs, {"baseline": money, "optimized": money, "change": money, "change_pct": lambda v: f"{v:+.1%}"})
    if baseline.violations:
        body += '<p class="note">Rules that today\'s allocation breaks: ' + escape("; ".join(baseline.violations)) + ".</p>"

    dcs = baseline.dc_summary[["dc", "customers", "units", "peak_utilization"]].merge(
        optimized.dc_summary[["dc", "open", "customers", "units", "peak_utilization", "freight_cost", "handling_cost", "fixed_cost"]],
        on="dc", suffixes=("_today", "_plan"))
    dcs["open"] = dcs["open"].map({True: "open", False: "closed"})
    body += "<h2>Distribution centers</h2><p class=\"note\">Peak utilization is the busiest week's units against weekly capacity.</p>"
    body += _table(dcs.rename(columns={"open": "in_plan"}), {
        "customers_today": _num, "customers_plan": _num, "units_today": _num, "units_plan": _num,
        "peak_utilization_today": pct, "peak_utilization_plan": pct, "freight_cost": money, "handling_cost": money, "fixed_cost": money})

    changes = c.lane_changes
    body += (f"<h2>Lanes that change</h2><p class=\"note\">The {min(top, len(changes))} largest of {len(changes)} changes. Cost is freight "
             "plus handling for the customer over all weeks. A single customer can cost more in the plan when it makes room for others "
             "or moves to meet the service rule.</p>")
    body += _table(changes.head(top)[["customer_id", "city", "state", "change", "from_dc", "to_dc", "from_mode", "to_mode",
                                      "transit_days_before", "transit_days_after", "cost_before", "cost_after", "saving"]],
                   {"cost_before": money, "cost_after": money, "saving": money, "transit_days_before": _num, "transit_days_after": _num},
                   empty="No customer changes DC or mode.")

    mix = baseline.mode_mix.merge(optimized.mode_mix, on="mode", how="outer", suffixes=("_today", "_plan")).fillna(0.0)
    body += "<h2>Modes</h2><p class=\"note\">Shipments are packages, LTL shipments or trucks over all weeks.</p>"
    body += _table(mix[["mode", "lanes_today", "lanes_plan", "shipments_today", "shipments_plan", "freight_cost_today", "freight_cost_plan"]],
                   {"lanes_today": _num, "lanes_plan": _num, "shipments_today": _num, "shipments_plan": _num,
                    "freight_cost_today": money, "freight_cost_plan": money})

    service = c.service.copy()
    body += "<h2>Service</h2>" + _table(service, {
        "baseline": lambda v: _num(v, 3), "optimized": lambda v: _num(v, 3)})

    if consolidation is not None and len(consolidation):
        s = consolidation_summary(consolidation)
        held = consolidation[consolidation["hold_days"] > 0]
        body += (f"<h2>Consolidation</h2><p>Holding orders for up to {int(consolidation['hold_days'].max())} working day(s) on "
                 f"{s['lanes_consolidated']} of {s['lanes']} lanes would cut scheduled shipments from {s['shipments_per_week_before']} to "
                 f"{s['shipments_per_week_after']} a week and freight by an estimated {money(s['saving'])} ({pct(s['saving_pct'], 1)}) on top "
                 "of the plan above, with days held plus transit still inside the service rule. The cost of holding goods and the "
                 "customer's view of fewer deliveries are not counted.</p>")
        body += _table(held.head(20)[["customer_id", "city", "dc", "shipments_per_week", "consolidated_shipments_per_week", "hold_days",
                                      "mode", "consolidated_mode", "order_to_delivery_days", "freight_before", "freight_after", "saving"]],
                       {"freight_before": money, "freight_after": money, "saving": money}, empty="No lane gains from consolidation.")
    if dc_loss is not None and len(dc_loss):
        body += ("<h2>If a DC is lost</h2><p class=\"note\">Each DC is removed in turn and the network re-planned. Extra cost leaves out "
                 "the lost DC's own fixed cost.</p>")
        body += _table(dc_loss, {"customers_affected": _num, "units_affected": _num, "extra_cost": money,
                                 "extra_cost_pct": lambda v: pct(v, 1), "share_within_service": lambda v: pct(v, 1),
                                 "average_transit_days": lambda v: _num(v, 2)})
    if fuel is not None and len(fuel):
        body += ("<h2>Fuel surcharge</h2><p class=\"note\">Total cost of today's allocation, of the plan left unchanged, and of a plan "
                 "optimized again at each surcharge.</p>")
        body += _table(fuel, {"fuel_surcharge": pct, "baseline_cost": money, "keep_plan_cost": money, "reoptimized_cost": money,
                              "saving_vs_baseline": money, "reoptimizing_gain": money, "customers_moved": _num, "dcs_open": _num})

    body += ('<h2>How to read this</h2><p class="caveat">Every figure is an estimate. Freight is computed from simple rate functions '
             "(distance, weight, a minimum charge, weight breaks and a fuel surcharge) that stand in for carrier contracts; they are "
             "illustrative, not real tariffs. Road miles are straight-line miles times 1.18. Transit days come from distance and mode. "
             "The plan assumes demand like the weeks supplied, and does not count the cost of changing a customer's DC, inventory, "
             "or carrier capacity. Check the largest changes against real quotes before acting on them.</p>")
    return _page("Distribution optimization report", "SupplyLens Distribution Optimizer", body)


def write_html(html: str, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
