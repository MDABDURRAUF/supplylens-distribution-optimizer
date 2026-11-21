# SupplyLens Distribution Optimizer

**Distribution cost optimization for a US distribution network: which distribution center should serve which
customer, by which mode, how often, and what that is estimated to save against how things are done today.**

Given a network and its demand, the optimizer does two things:

1. **Costs today's allocation and the cheapest allocation that obeys your rules** with the same rate functions on
   the same demand, so the two are compared like for like.
2. **Explains the difference**: the cost breakdown, every lane that changes, what happens to service, and how the
   answer holds up if fuel, demand or the network itself changes.

Every saving it reports is an estimate from the rates and demand supplied, not a realized saving.

The included network is simulated, and the freight rate functions are illustrative, not real carrier tariffs.
Results on them show that the method works; they are not findings about real companies or real freight prices.

## What it does

| Question | Where | Function |
|---|---|---|
| What would the cheapest allocation save against today's? | **Savings overview** | `optimize`, `baseline_plan`, `compare` |
| Which DC serves which customer, before and after? | **Network map** | `Result.lanes` |
| Which customers change, and what does each change save? | **Assignment explorer** | `Comparison.lane_changes`, `lane_options` |
| Which mode on each lane, and is it worth shipping less often? | **Modes and consolidation** | `baseline_with_best_modes`, `consolidation_analysis` |
| Which DCs should operate at all? | **Scenarios** | `open_close`, `dc_count_table` |
| What if fuel rises, demand grows or a DC is lost? | **Scenarios** | `fuel_sensitivity`, `demand_growth`, `dc_loss_table` |
| Why is there no plan, and which rule should give? | every view | `diagnose` |
| What does any given allocation cost, and which rules does it break? | | `cost_plan` |

Rules you can set: the longest transit time (the service rule), one DC per customer or splitting allowed, whether
the plan may close DCs, which modes may be used, how much of each DC's capacity may be used, and the fuel surcharge.

Outputs: the recommended assignment and the list of lanes that change as CSV, a single-file HTML optimization
report with a plain-language summary, and a command line for all of it.

## Results on the simulated data

Six distribution centers (Los Angeles, Dallas, Chicago, Atlanta, Newark, Memphis) serving 150 customer locations
in US metro areas for 13 weeks: 906,899 units of four product families. Today's allocation is simulated as habit
rather than choice. Rules: delivery within 4 transit days, every DC stays open, fuel surcharge 20%.

| Cost over 13 weeks | Today | Recommended | Change |
|---|---|---|---|
| Freight | $1,955,213 | $1,527,535 | -$427,678 |
| Handling | $444,641 | $454,729 | +$10,089 |
| Fixed DC cost | $669,500 | $669,500 | $0 |
| **Total** | **$3,069,354** | **$2,651,764** | **-$417,590 (-13.6%)** |

- Estimated saving: $417,590 over 13 weeks, about $32,122 a week. 78 of 150 customers change: 70 move to another
  DC and 17 change mode. Five customers are split between two DCs to fit capacity.
- Service improves rather than pays for the saving: units delivered within the 4-day rule go from 93.3% to 100%,
  average transit from 2.32 to 1.93 days, and the average unit travels 484 road miles instead of 598.
- With one DC per customer the estimated saving is $395,211 (12.9%).
- Mode choice alone, with no customer changing DC, is worth an estimated $59,213.
- Free to close DCs, the plan closes Memphis and the estimated saving rises to $473,293 (15.4%): $55,703 better
  than keeping all six open. Closing costs are not in the model.
- Consolidation on top of the plan: holding orders up to 2 working days consolidates 58 of 155 lanes, cuts
  scheduled shipments from 384 to 290 a week and freight by an estimated $101,202 (6.6%), before any cost of
  holding goods. A 1-day hold is worth only $5,595.
- Fuel: every 10 points of surcharge adds about $162,934 to today's allocation and $127,295 to the plan, so the
  estimated saving grows from $346,402 at no surcharge to $524,509 at 50%. Re-planning for a different fuel price
  adds at most $92, because the surcharge raises every lane's freight by the same percentage.
- Growth: the network can serve up to 30% more demand within the rules; at 40% the busiest week (111,272 units)
  exceeds the capacity of all DCs together (107,000), and the tool says so and names DC capacity as the rule to relax.
- Loss of a DC: Memphis is the cheapest to lose (+$48,297, +1.8%). Chicago cannot be lost: the others lack the
  capacity. Losing Los Angeles doubles the cost (+$2,669,976) because the 4-day rule can then only be met in the
  West by truckload; with a 5-day rule the same loss costs +$368,896 (+14.0%).

These numbers come from a simulated network and made-up rate functions. The size of a saving on a real network
depends on how far today's allocation is from the best one and on real contract rates.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-distribution-optimizer.git
cd supplylens-distribution-optimizer
pip install -e ".[app,dev]"
streamlit run app.py
```

Command line:

```bash
distribution-optimizer sample --out network
distribution-optimizer optimize --data network --out assignment.csv --changes lane_changes.csv
distribution-optimizer optimize --data network --single-sourcing --choose-dcs --max-transit-days 5 --fuel 0.25
distribution-optimizer scenario dc-loss --data network
distribution-optimizer scenario fuel --data network --values 0 0.2 0.4
distribution-optimizer scenario growth --data network
distribution-optimizer scenario open-close --data network
distribution-optimizer scenario consolidation --data network --hold-days 2
distribution-optimizer report --data network --out optimization_report.html
```

Without `--data`, every command uses the simulated sample network. Sample outputs are in [examples](examples).

Python:

```python
import distribution_optimizer as do

network = do.load_network("network")                 # or do.generate_sample_network()
settings = do.Settings(max_transit_days=4, single_sourcing=False, choose_dcs=False)

today = do.baseline_plan(network, settings)          # today's allocation, costed as recorded
plan = do.optimize(network, settings)                # cheapest allocation that obeys the rules
if not plan.ok:
    print(plan.message, do.diagnose(network, settings))   # which rule to relax

comparison = do.compare(today, plan)
print(do.savings_headline(comparison, today, plan, settings))
print(comparison.costs)                              # freight, handling, fixed: today against the plan
print(comparison.lane_changes.head())                # who moves, from where to where, and the saving
plan.lanes.to_csv("assignment.csv", index=False)

print(do.consolidation_summary(do.consolidation_analysis(network, plan, settings, max_hold_days=2)))
print(do.dc_loss_table(network, settings))
print(do.fuel_sensitivity(network, settings.with_fuel(0.20), surcharges=(0.0, 0.2, 0.4)))
```

## Input data

A folder with five CSV files.

| File | Column | Required | Meaning |
|---|---|---|---|
| `dcs.csv` | `dc` | yes | Name of the distribution center (unique) |
| | `state`, `latitude`, `longitude` | yes | Two-letter US state code and location in degrees |
| | `capacity_units_per_week` | yes | Most units the DC can ship in one week, all product families together |
| | `handling_cost_per_unit` | yes | Cost of picking, packing and loading one unit, in dollars |
| | `fixed_cost_per_week` | no | Cost of keeping the DC open for a week, whatever it ships (default 0) |
| `customers.csv` | `customer_id` | yes | Unique identifier |
| | `state`, `latitude`, `longitude` | yes | Two-letter US state code and location in degrees |
| | `city` | no | City or site name, for display |
| | `shipments_per_week` | no | Separate shipments the customer receives in a week today (default 1) |
| `products.csv` | `product_family`, `weight_lbs_per_unit` | yes | Family name and shipping weight of one unit |
| `demand.csv` | `week`, `customer_id`, `product_family`, `units` | yes | Units ordered, by week |
| `baseline.csv` | `customer_id`, `dc` | yes | Which DC serves the customer today |
| | `mode` | no | `parcel`, `LTL` or `truckload`; blank means the cheapest mode is assumed |
| | `share` | no | Share of the customer's demand from this DC (default 1; a customer's rows must add to 1) |

Files are validated on load against each other; every problem is reported with the number of rows affected, and
nothing is dropped silently. Freight rates are not a file: they are the parameters of `RateCard`, passed in
`Settings(rates=...)`.

## How it works

**Costing a lane.** A lane is one DC serving one customer. The customer receives `shipments_per_week` shipments
each week, each carrying that week's pounds divided equally. Freight is the mode's charge for each shipment, added
over all weeks; handling is the DC's cost per unit; fixed cost is paid by every open DC whatever it ships. Road
miles are great-circle miles times 1.18.

**Rate functions** (illustrative, in `rates.py`; all parameters can be replaced):

| Mode | Charge for one shipment, before fuel | Limit | Transit days |
|---|---|---|---|
| Parcel | `max($12, $9 + lb x ($0.28 + $0.00045 x miles))` per package | 150 lb per package | miles / 400, rounded up |
| LTL | `max($110, lb / 100 x ($14 + $0.035 x miles) x discount)`; discount 1.00, 0.88, 0.76, 0.64, 0.54, 0.46 from 0, 500, 1,000, 2,000, 5,000 and 10,000 lb | 20,000 lb | 1 + miles / 450, rounded up |
| Truckload | `max($650, $2.30 x miles)` per truck | 42,000 lb per truck | miles / 500, rounded up |

An LTL shipment is billed at the start of the next weight break when that is cheaper, so a heavier shipment never
costs less. A shipment over the limit is sent as the fewest equal shipments that fit. The fuel surcharge is a
percentage added to every charge.

**Assignment.** `optimize` chooses, for every customer, the share of its demand served by each DC, to minimize
freight plus handling (plus fixed cost when it may close DCs), subject to: all demand served; each DC within its
capacity in every week; lanes used only where some allowed mode delivers within the service rule; optionally one
DC per customer. Each lane uses the cheapest allowed mode that meets the rule. This is a linear program, or a
mixed-integer program with single sourcing or DC choices, solved with `scipy.optimize.milp` (HiGHS).

**Splitting a customer** means a share of its shipments comes from each DC, with every shipment at its usual
size, so cost and DC load scale with the share. Shares are not rounded to whole shipments.

**Comparison with today.** `baseline_plan` costs the `baseline` table exactly as recorded, with its own DCs and
modes, through the same function (`cost_plan`) that costs the recommended plan. Rules that today's allocation
breaks are listed in `violations`. The saving is baseline total minus plan total.

**Consolidation.** Holding orders up to `h` working days means shipping at most once every `h + 1` working days,
so at most `ceil(5 / (h + 1))` shipments a week. For each lane the analysis picks the hold (and mode) with the
lowest freight, among those where days held plus transit days stay within the service rule.

**Scenarios.** Fuel: today's allocation, the plan left unchanged and a re-optimized plan at each surcharge.
Growth: every demand line scaled and the network re-planned. Loss of a DC: each DC removed in turn and the rest
re-planned; the extra cost leaves out the lost DC's own fixed cost. Open or close: the plan chooses which DCs
operate, and `dc_count_table` gives the best network for each number of DCs.

**When no plan exists**, the result says why (for example which customers cannot be reached in time, or that the
busiest week exceeds total capacity), and `diagnose` re-solves with one rule relaxed at a time to name the rules
whose removal alone would make a plan possible, including the shortest transit limit that works.

## Limits

- The sample network, its demand and its "as-is" allocation are simulated.
- The rate functions are illustrative. They are not any carrier's tariff and include no accessorials, lane-specific
  pricing or contract discounts. Replace the `RateCard` values before acting on a number.
- Every saving is an estimate against a modelled baseline, not a realized saving. It leaves out the cost of moving
  a customer to another DC, inventory, and closing or opening costs.
- Demand is taken as given, week by week: no forecast and no uncertainty. Shipments are the same size within a week.
- Distances are great-circle miles times 1.18, not routed miles; metro coordinates are approximate. Transit days
  depend only on distance and mode.
- A DC's capacity is one number of units per week for all product families; there are no dock, labour or
  inventory limits, and no carrier capacity limits.
- One mode per lane for the whole period; no multi-stop routes, pooling between customers or DC-to-DC transfers.
- With one DC per customer and nearly full DCs, the solver can reach its time limit; the result then has status
  `feasible` (best plan found), not `optimal`.
- Consolidation counts freight only, not the cost of holding goods or what customers think of fewer deliveries.

## Tests

```bash
pytest
```

Twenty tests: rate functions and transit days checked against hand-worked charges; tiny networks whose best
allocation can be stated by hand (capacity and single sourcing, nearest DC, opening and closing a DC, mode choice
under the service rule, consolidation); infeasible cases and the rule to relax; like-for-like costing; every rule
checked on the optimized sample; scenarios; validation; the report and the command line.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): supply chain disruption analytics on purchase orders
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses
- [SupplyLens Network Simulator](https://github.com/MDABDURRAUF/supplylens-network-simulator): disruption scenarios for service and goods networks

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
