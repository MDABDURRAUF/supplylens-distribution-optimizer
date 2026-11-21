"""Command line.

    distribution-optimizer sample --out network
    distribution-optimizer optimize --data network --out assignment.csv --changes lane_changes.csv
    distribution-optimizer scenario dc-loss --data network
    distribution-optimizer scenario fuel --data network --values 0 0.2 0.4
    distribution-optimizer report --data network --out report.html

Without ``--data`` every command uses the simulated sample network.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .consolidation import consolidation_analysis, consolidation_summary
from .data import SchemaError, generate_sample_network, load_network
from .optimizer import Settings, baseline_plan, compare, optimize
from .rates import MODES, RateCard
from .report import build_report, money, rules_text, savings_headline, write_html
from .scenarios import dc_count_table, dc_loss_table, demand_growth, diagnose, fuel_sensitivity, open_close

SCENARIOS = ("fuel", "growth", "dc-loss", "open-close", "consolidation")


def _show(df: pd.DataFrame, formatters: dict | None = None) -> None:
    """Print a table; empty cells (no plan) are shown as a dash."""
    def safe(fmt):
        return lambda v: "-" if pd.isna(v) else fmt(v)
    formatters = {k: safe(f) for k, f in (formatters or {}).items() if k in df.columns}
    with pd.option_context("display.width", 240, "display.max_columns", None, "display.max_colwidth", 90):
        print(df.to_string(index=False, formatters=formatters, na_rep="-"))


def _notes(plan) -> None:
    if plan.status != "optimal":
        print(f"\nNote: {plan.message}")
    if plan.split_customers:
        print(f"\nNote: {plan.split_customers} customer(s) are split between DCs: a share of their shipments comes from each.")


def _rules(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="folder with dcs.csv, customers.csv, products.csv, demand.csv and baseline.csv (default: simulated sample)")
    p.add_argument("--max-transit-days", type=int, default=4, help="service rule in days; 0 means no rule (default 4)")
    p.add_argument("--single-sourcing", action="store_true", help="serve every customer from exactly one DC")
    p.add_argument("--choose-dcs", action="store_true", help="let the plan close DCs to save their fixed cost")
    p.add_argument("--fuel", type=float, default=RateCard().fuel_surcharge, help="fuel surcharge, e.g. 0.20 for 20%% (default 0.20)")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="modes the plan may use")
    p.add_argument("--capacity-use", type=float, default=1.0, help="share of DC capacity the plan may use (default 1.0)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distribution-optimizer", description="Distribution cost optimization against today's allocation.")
    p.add_argument("--version", action="version", version=f"distribution-optimizer {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write a simulated US network as five CSV files")
    s.add_argument("--out", default="network", help="folder to write into")
    s.add_argument("--customers", type=int, default=150)
    s.add_argument("--weeks", type=int, default=13)
    s.add_argument("--seed", type=int, default=7)

    o = sub.add_parser("optimize", help="find the cheapest allocation and compare it with today's")
    _rules(o)
    o.add_argument("--out", help="write the recommended assignment to this CSV file")
    o.add_argument("--changes", help="write the list of lanes that change to this CSV file")

    sc = sub.add_parser("scenario", help="what-if tests on the optimized network")
    sc.add_argument("kind", choices=SCENARIOS)
    _rules(sc)
    sc.add_argument("--values", type=float, nargs="+", help="fuel surcharges or growth rates to test, e.g. 0 0.2 0.4")
    sc.add_argument("--hold-days", type=int, default=2, help="longest hold for consolidation, in working days (default 2)")
    sc.add_argument("--out", help="write the scenario table to this CSV file")

    r = sub.add_parser("report", help="write the optimization report as one HTML file")
    _rules(r)
    r.add_argument("--out", default="optimization_report.html")
    r.add_argument("--hold-days", type=int, default=2, help="longest hold for the consolidation section (default 2)")
    r.add_argument("--no-scenarios", action="store_true", help="leave out the DC-loss and fuel sections")
    return p


def _setup(args):
    network = load_network(args.data) if args.data else generate_sample_network()
    if not args.data:
        print("Using the simulated sample network. Not real data.")
    settings = Settings(max_transit_days=args.max_transit_days or None, single_sourcing=args.single_sourcing, choose_dcs=args.choose_dcs,
                        modes=tuple(args.modes), max_capacity_use=args.capacity_use, rates=RateCard(fuel_surcharge=args.fuel))
    return network, settings


def _plan(network, settings):
    """Today's allocation, the optimized one and their comparison; raises with advice when no plan exists."""
    baseline, plan = baseline_plan(network, settings), optimize(network, settings)
    if not plan.ok:
        fixes = diagnose(network, settings)
        advice = f" Relaxing {', or '.join(fixes)} would make a plan possible." if fixes else " No single rule is to blame."
        raise RuntimeError(plan.message + advice)
    return baseline, plan, compare(baseline, plan)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    usd, pc, whole = money, "{:.1%}".format, "{:,.0f}".format
    try:
        if args.command == "sample":
            network = generate_sample_network(customers=args.customers, weeks=args.weeks, seed=args.seed)
            folder = network.save(args.out)
            print(f"Wrote a simulated network to {folder}: {len(network.dcs)} DCs, {len(network.customers)} customers, "
                  f"{len(network.weeks)} weeks, {len(network.demand):,} demand lines. Not real data.")
            return 0
        network, settings = _setup(args)
        print(f"Rules: {rules_text(settings)}")
        if args.command == "optimize":
            baseline, plan, c = _plan(network, settings)
            print(savings_headline(c, baseline, plan, settings))
            print()
            _show(c.costs, {"baseline": usd, "optimized": usd, "change": usd, "change_pct": "{:+.1%}".format})
            print()
            _show(plan.dc_summary[["dc", "open", "customers", "units", "peak_utilization", "freight_cost", "handling_cost", "fixed_cost"]],
                  {"units": "{:,.0f}".format, "peak_utilization": pc, "freight_cost": usd, "handling_cost": usd, "fixed_cost": usd})
            if len(c.lane_changes):
                print("\nLargest changes:")
                _show(c.lane_changes.head(10)[["customer_id", "city", "change", "from_dc", "to_dc", "from_mode", "to_mode", "saving"]], {"saving": usd})
            _notes(plan)
            if args.out:
                plan.lanes.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
            if args.changes:
                c.lane_changes.to_csv(args.changes, index=False)
                print(f"Wrote {args.changes}")
        elif args.command == "scenario":
            if args.kind == "fuel":
                table = fuel_sensitivity(network, settings, args.values or (0.0, 0.10, 0.20, 0.30, 0.40, 0.50))
                _show(table, {"fuel_surcharge": "{:.0%}".format, **{k: usd for k in table.columns if k.endswith(("_cost", "_baseline", "_gain"))}})
                print("\nkeep_plan_cost: the plan optimized at the current surcharge, left unchanged. reoptimized_cost: planned again at that surcharge.")
            elif args.kind == "growth":
                table = demand_growth(network, settings, args.values or (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40))
                _show(table.drop(columns="note"), {"growth": "{:+.0%}".format, "units": whole, "total_cost": usd, "freight_cost": usd,
                                                   "handling_cost": usd, "fixed_cost": usd, "cost_per_unit": "${:.3f}".format,
                                                   "peak_utilization": pc, "dcs_full": whole, "split_customers": whole, "dcs_open": whole})
                for row in table[table["note"].fillna("") != ""].itertuples():
                    print(f"\n{row.growth:+.0%}: {row.note}")
            elif args.kind == "dc-loss":
                table = dc_loss_table(network, settings)
                _show(table, {"units_affected": whole, "extra_cost": usd, "extra_cost_pct": "{:+.1%}".format,
                              "share_within_service": pc, "average_transit_days": "{:.2f}".format})
            elif args.kind == "open-close":
                out = open_close(network, settings)
                if not out["chosen"].ok:
                    raise RuntimeError(out["chosen"].message)
                closed = ", ".join(out["closed"]) or "none"
                print(f"Free to choose, the plan keeps {len(out['chosen'].open_dcs)} of {len(network.dcs)} DCs open (closed: {closed}). "
                      f"Estimated saving against keeping all open: {usd(out['saving'])} over {out['chosen'].weeks} weeks. "
                      "Closing costs and lease terms are not counted.\n")
                table = dc_count_table(network, settings)
                _show(table, {"total_cost": usd, "freight_cost": usd, "handling_cost": usd, "fixed_cost": usd, "share_within_service": pc,
                              "average_transit_days": "{:.2f}".format})
            else:
                _, plan, _ = _plan(network, settings)
                table = consolidation_analysis(network, plan, settings, args.hold_days)
                s = consolidation_summary(table)
                print(f"Holding orders up to {args.hold_days} working day(s): {s['lanes_consolidated']} of {s['lanes']} lanes consolidate, "
                      f"scheduled shipments go from {s['shipments_per_week_before']} to {s['shipments_per_week_after']} a week, and freight "
                      f"falls by an estimated {usd(s['saving'])} ({s['saving_pct']:.1%}) on the optimized plan. Holding costs are not counted.\n")
                _show(table[table["hold_days"] > 0].head(15)[["customer_id", "city", "dc", "shipments_per_week", "consolidated_shipments_per_week",
                                                             "hold_days", "mode", "consolidated_mode", "order_to_delivery_days", "saving"]], {"saving": usd})
            if args.out:
                table.to_csv(args.out, index=False)
                print(f"\nWrote {args.out}")
        elif args.command == "report":
            baseline, plan, c = _plan(network, settings)
            extra = {} if args.no_scenarios else {"dc_loss": dc_loss_table(network, settings), "fuel": fuel_sensitivity(network, settings)}
            html = build_report(baseline, plan, c, settings, consolidation=consolidation_analysis(network, plan, settings, args.hold_days),
                                simulated=not args.data, **extra)
            print(savings_headline(c, baseline, plan, settings))
            print(f"Wrote {write_html(html, args.out)}")
    except (SchemaError, ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
