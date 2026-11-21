"""United States reference data: Census regions, federal holidays and metro locations."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# U.S. Census Bureau regions (50 states and the District of Columbia).
CENSUS_REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_REGION = {state: region for region, states in CENSUS_REGIONS.items() for state in states}

# Freight hubs used by the sample data: metro -> (state, latitude, longitude). Coordinates are approximate.
METROS = {
    "Los Angeles": ("CA", 34.05, -118.24), "Seattle": ("WA", 47.61, -122.33), "Phoenix": ("AZ", 33.45, -112.07),
    "Salt Lake City": ("UT", 40.76, -111.89), "Denver": ("CO", 39.74, -104.99), "Dallas": ("TX", 32.78, -96.80),
    "Houston": ("TX", 29.76, -95.37), "Kansas City": ("MO", 39.10, -94.58), "Minneapolis": ("MN", 44.98, -93.27),
    "Chicago": ("IL", 41.88, -87.63), "Detroit": ("MI", 42.33, -83.05), "Columbus": ("OH", 39.96, -83.00),
    "Memphis": ("TN", 35.15, -90.05), "Nashville": ("TN", 36.16, -86.78), "Atlanta": ("GA", 33.75, -84.39),
    "Miami": ("FL", 25.76, -80.19), "Charlotte": ("NC", 35.23, -80.84), "Philadelphia": ("PA", 39.95, -75.17),
    "Newark": ("NJ", 40.74, -74.17), "Boston": ("MA", 42.36, -71.06),
}

# Further metro areas in the contiguous United States, used as customer locations by the sample
# network: metro -> (state, latitude, longitude). Coordinates are approximate (two decimals, city
# centre) and are good enough for distance estimates, not for routing. Where two metros share a
# name the state is added in brackets.
MORE_METROS = {
    # West
    "San Diego": ("CA", 32.72, -117.16), "San Francisco": ("CA", 37.77, -122.42), "San Jose": ("CA", 37.34, -121.89),
    "Sacramento": ("CA", 38.58, -121.49), "Fresno": ("CA", 36.74, -119.79), "Bakersfield": ("CA", 35.37, -119.02),
    "Riverside": ("CA", 33.95, -117.40), "Stockton": ("CA", 37.96, -121.29), "Oakland": ("CA", 37.80, -122.27),
    "Santa Rosa": ("CA", 38.44, -122.71), "Modesto": ("CA", 37.64, -120.99), "Oxnard": ("CA", 34.20, -119.18),
    "Portland (OR)": ("OR", 45.52, -122.68), "Eugene": ("OR", 44.05, -123.09), "Salem": ("OR", 44.94, -123.04),
    "Medford": ("OR", 42.33, -122.87), "Spokane": ("WA", 47.66, -117.43), "Tacoma": ("WA", 47.25, -122.44),
    "Yakima": ("WA", 46.60, -120.51), "Boise": ("ID", 43.62, -116.20), "Idaho Falls": ("ID", 43.49, -112.03),
    "Las Vegas": ("NV", 36.17, -115.14), "Reno": ("NV", 39.53, -119.81), "Tucson": ("AZ", 32.22, -110.97),
    "Flagstaff": ("AZ", 35.20, -111.65), "Albuquerque": ("NM", 35.08, -106.65), "Las Cruces": ("NM", 32.32, -106.76),
    "Colorado Springs": ("CO", 38.83, -104.82), "Fort Collins": ("CO", 40.59, -105.08),
    "Grand Junction": ("CO", 39.06, -108.55), "Billings": ("MT", 45.78, -108.50), "Missoula": ("MT", 46.87, -113.99),
    "Cheyenne": ("WY", 41.14, -104.82), "Provo": ("UT", 40.23, -111.66),
    # Midwest
    "Indianapolis": ("IN", 39.77, -86.16), "Milwaukee": ("WI", 43.04, -87.91), "St. Louis": ("MO", 38.63, -90.20),
    "Cleveland": ("OH", 41.50, -81.69), "Cincinnati": ("OH", 39.10, -84.51), "Omaha": ("NE", 41.26, -95.93),
    "Des Moines": ("IA", 41.59, -93.62), "Wichita": ("KS", 37.69, -97.34), "Madison": ("WI", 43.07, -89.40),
    "Grand Rapids": ("MI", 42.96, -85.67), "Toledo": ("OH", 41.65, -83.54), "Dayton": ("OH", 39.76, -84.19),
    "Akron": ("OH", 41.08, -81.52), "Fort Wayne": ("IN", 41.08, -85.14), "Lincoln": ("NE", 40.81, -96.70),
    "Sioux Falls": ("SD", 43.55, -96.73), "Fargo": ("ND", 46.88, -96.79), "Springfield (MO)": ("MO", 37.21, -93.29),
    "Peoria": ("IL", 40.69, -89.59), "Rockford": ("IL", 42.27, -89.09), "Green Bay": ("WI", 44.51, -88.01),
    "Cedar Rapids": ("IA", 41.98, -91.67), "Davenport": ("IA", 41.52, -90.58), "Lansing": ("MI", 42.73, -84.56),
    "Evansville": ("IN", 37.97, -87.56), "South Bend": ("IN", 41.68, -86.25), "Topeka": ("KS", 39.05, -95.68),
    "Duluth": ("MN", 46.79, -92.10), "Rochester (MN)": ("MN", 44.02, -92.47), "Bismarck": ("ND", 46.81, -100.78),
    "Rapid City": ("SD", 44.08, -103.23), "Youngstown": ("OH", 41.10, -80.65), "Flint": ("MI", 43.01, -83.69),
    "Springfield (IL)": ("IL", 39.80, -89.64), "Kalamazoo": ("MI", 42.29, -85.59),
    # South
    "San Antonio": ("TX", 29.42, -98.49), "Austin": ("TX", 30.27, -97.74), "Fort Worth": ("TX", 32.76, -97.33),
    "El Paso": ("TX", 31.76, -106.49), "Corpus Christi": ("TX", 27.80, -97.40), "Lubbock": ("TX", 33.58, -101.86),
    "Amarillo": ("TX", 35.22, -101.83), "McAllen": ("TX", 26.20, -98.23), "Midland": ("TX", 32.00, -102.08),
    "Oklahoma City": ("OK", 35.47, -97.52), "Tulsa": ("OK", 36.15, -95.99), "New Orleans": ("LA", 29.95, -90.07),
    "Baton Rouge": ("LA", 30.45, -91.19), "Shreveport": ("LA", 32.53, -93.75), "Lafayette": ("LA", 30.22, -92.02),
    "Little Rock": ("AR", 34.75, -92.29), "Fayetteville": ("AR", 36.06, -94.16), "Jackson": ("MS", 32.30, -90.18),
    "Gulfport": ("MS", 30.37, -89.09), "Birmingham": ("AL", 33.52, -86.80), "Mobile": ("AL", 30.69, -88.04),
    "Huntsville": ("AL", 34.73, -86.59), "Montgomery": ("AL", 32.37, -86.30), "Louisville": ("KY", 38.25, -85.76),
    "Lexington": ("KY", 38.04, -84.50), "Knoxville": ("TN", 35.96, -83.92), "Chattanooga": ("TN", 35.05, -85.31),
    "Jacksonville": ("FL", 30.33, -81.66), "Tampa": ("FL", 27.95, -82.46), "Orlando": ("FL", 28.54, -81.38),
    "Fort Myers": ("FL", 26.64, -81.87), "Tallahassee": ("FL", 30.44, -84.28), "Pensacola": ("FL", 30.42, -87.22),
    "Savannah": ("GA", 32.08, -81.09), "Augusta": ("GA", 33.47, -81.97), "Macon": ("GA", 32.84, -83.63),
    "Columbia": ("SC", 34.00, -81.03), "Charleston (SC)": ("SC", 32.78, -79.93), "Greenville": ("SC", 34.85, -82.40),
    "Raleigh": ("NC", 35.78, -78.64), "Greensboro": ("NC", 36.07, -79.79), "Winston-Salem": ("NC", 36.10, -80.24),
    "Asheville": ("NC", 35.60, -82.55), "Wilmington (NC)": ("NC", 34.23, -77.94), "Richmond": ("VA", 37.54, -77.44),
    "Norfolk": ("VA", 36.85, -76.29), "Roanoke": ("VA", 37.27, -79.94), "Washington": ("DC", 38.91, -77.04),
    "Baltimore": ("MD", 39.29, -76.61), "Wilmington (DE)": ("DE", 39.75, -75.55), "Charleston (WV)": ("WV", 38.35, -81.63),
    # Northeast
    "New York": ("NY", 40.71, -74.01), "Pittsburgh": ("PA", 40.44, -80.00), "Buffalo": ("NY", 42.89, -78.88),
    "Rochester (NY)": ("NY", 43.16, -77.61), "Syracuse": ("NY", 43.05, -76.15), "Albany": ("NY", 42.65, -73.76),
    "Binghamton": ("NY", 42.10, -75.91), "Hartford": ("CT", 41.76, -72.69), "New Haven": ("CT", 41.31, -72.92),
    "Stamford": ("CT", 41.05, -73.54), "Providence": ("RI", 41.82, -71.41), "Worcester": ("MA", 42.26, -71.80),
    "Springfield (MA)": ("MA", 42.10, -72.59), "Manchester": ("NH", 42.99, -71.46), "Portland (ME)": ("ME", 43.66, -70.26),
    "Bangor": ("ME", 44.80, -68.77), "Burlington": ("VT", 44.48, -73.21), "Allentown": ("PA", 40.60, -75.47),
    "Harrisburg": ("PA", 40.27, -76.88), "Scranton": ("PA", 41.41, -75.66), "Erie": ("PA", 42.13, -80.09),
    "Lancaster": ("PA", 40.04, -76.31), "Trenton": ("NJ", 40.22, -74.76), "Atlantic City": ("NJ", 39.36, -74.42),
}
ALL_METROS = {**METROS, **MORE_METROS}

# Road miles are estimated as great-circle miles times this factor (roads are not straight lines).
ROAD_FACTOR = 1.18


def road_miles(lat1, lon1, lat2, lon2):
    """Estimated road miles between points given in degrees: great-circle miles times ``ROAD_FACTOR``."""
    return great_circle_miles(lat1, lon1, lat2, lon2) * ROAD_FACTOR


def region_of(state: str) -> str:
    """Census region of a two-letter state code."""
    try:
        return STATE_REGION[str(state).strip().upper()]
    except KeyError:
        raise ValueError(f"'{state}' is not a US state code") from None


def great_circle_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between points given in degrees (arrays allowed)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th given weekday of a month (Monday = 0); n = -1 means the last one."""
    if n > 0:
        first = dt.date(year, month, 1)
        return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def federal_holidays(years) -> dict[dt.date, str]:
    """US federal holidays for the given year or years, on their actual dates: ``{date: name}``."""
    years = [years] if isinstance(years, (int, np.integer)) else list(years)
    out = {}
    for y in years:
        y = int(y)
        out.update({
            dt.date(y, 1, 1): "New Year's Day",
            _nth_weekday(y, 1, 0, 3): "Birthday of Martin Luther King, Jr.",
            _nth_weekday(y, 2, 0, 3): "Washington's Birthday",
            _nth_weekday(y, 5, 0, -1): "Memorial Day",
            dt.date(y, 6, 19): "Juneteenth National Independence Day",
            dt.date(y, 7, 4): "Independence Day",
            _nth_weekday(y, 9, 0, 1): "Labor Day",
            _nth_weekday(y, 10, 0, 2): "Columbus Day",
            dt.date(y, 11, 11): "Veterans Day",
            _nth_weekday(y, 11, 3, 4): "Thanksgiving Day",
            dt.date(y, 12, 25): "Christmas Day",
        })
    return out


def holidays_between(start: pd.Series, end: pd.Series) -> np.ndarray:
    """For each row, the number of federal holidays from ``start`` to ``end`` inclusive."""
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    years = range(int(start.dt.year.min()), int(end.dt.year.max()) + 1)
    days = np.sort(np.array([np.datetime64(d) for d in federal_holidays(years)], dtype="datetime64[D]"))
    lo = np.searchsorted(days, start.to_numpy().astype("datetime64[D]"), side="left")
    hi = np.searchsorted(days, end.to_numpy().astype("datetime64[D]"), side="right")
    return hi - lo


def is_peak_season(dates: pd.Series) -> np.ndarray:
    """Whether a date falls in the year-end shipping peak (15 November to 31 December)."""
    dates = pd.to_datetime(dates)
    return (((dates.dt.month == 11) & (dates.dt.day >= 15)) | (dates.dt.month == 12)).to_numpy()
