"""SupplyLens Distribution Optimizer: distribution cost optimization for a US distribution network."""

__version__ = "0.1.0"

from . import scenarios  # noqa: E402
from .consolidation import consolidation_analysis, consolidation_by_window, consolidation_summary, shipments_after_hold  # noqa: E402
from .data import COLUMN_HELP, Network, SchemaError, generate_sample_network, load_network, validate_network  # noqa: E402
from .optimizer import (  # noqa: E402
    Comparison, Result, Settings, baseline_plan, baseline_with_best_modes, compare, cost_plan, lane_options, optimize,
)
from .rates import MODES, RateCard, rate_curve, shipment_cost, shipment_count, transit_days  # noqa: E402
from .report import build_report, rules_text, savings_headline, write_html  # noqa: E402
from .scenarios import (  # noqa: E402
    dc_count_table, dc_loss, dc_loss_table, demand_growth, diagnose, fuel_sensitivity, minimum_transit_days, open_close,
)
from .us import ALL_METROS, CENSUS_REGIONS, METROS, great_circle_miles, region_of, road_miles  # noqa: E402

__all__ = [
    "ALL_METROS", "CENSUS_REGIONS", "COLUMN_HELP", "Comparison", "METROS", "MODES", "Network", "RateCard", "Result", "SchemaError",
    "Settings", "baseline_plan", "baseline_with_best_modes", "build_report", "compare", "consolidation_analysis",
    "consolidation_by_window", "consolidation_summary", "cost_plan", "dc_count_table", "dc_loss", "dc_loss_table", "demand_growth",
    "diagnose", "fuel_sensitivity", "generate_sample_network", "great_circle_miles", "lane_options", "load_network",
    "minimum_transit_days", "open_close", "optimize", "rate_curve", "region_of", "road_miles", "rules_text", "savings_headline",
    "scenarios", "shipment_cost", "shipment_count", "shipments_after_hold", "transit_days", "validate_network", "write_html",
]
