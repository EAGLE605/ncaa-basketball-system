"""
Shared pytest fixtures for NCAAB Men's Basketball system tests.

All external API calls (The Odds API, BartTorvik, ESPN) are mocked.
Use --mock flag (implicit here) — never call paid APIs in test suite.
"""

import pytest


# ---------------------------------------------------------------------------
# Sample team data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def duke_stats():
    return {"AdjOE": 122.35, "AdjDE": 85.66, "AdjEM": 36.69, "AdjT": 67.5}


@pytest.fixture
def michigan_stats():
    return {"AdjOE": 120.27, "AdjDE": 83.66, "AdjEM": 36.61, "AdjT": 65.2}


@pytest.fixture
def arizona_stats():
    return {"AdjOE": 119.12, "AdjDE": 83.58, "AdjEM": 35.54, "AdjT": 69.1}


@pytest.fixture
def florida_stats():
    return {"AdjOE": 115.60, "AdjDE": 81.78, "AdjEM": 33.82, "AdjT": 68.3}


@pytest.fixture
def small_bracket():
    """Minimal 8-team (2-region, 4 per region) bracket for fast simulation tests."""
    return {
        "East":    ["Duke",     "seed16E", "seed8E",  "seed9E",
                    "seed5E",   "seed12E", "seed4E",  "seed13E",
                    "seed6E",   "seed11E", "seed3E",  "seed14E",
                    "seed7E",   "seed10E", "seed2E",  "seed15E"],
        "West":    ["Arizona",  "seed16W", "seed8W",  "seed9W",
                    "seed5W",   "seed12W", "seed4W",  "seed13W",
                    "seed6W",   "seed11W", "seed3W",  "seed14W",
                    "seed7W",   "seed10W", "seed2W",  "seed15W"],
        "Midwest": ["Michigan", "seed16M", "seed8M",  "seed9M",
                    "seed5M",   "seed12M", "seed4M",  "seed13M",
                    "seed6M",   "seed11M", "seed3M",  "seed14M",
                    "seed7M",   "seed10M", "seed2M",  "seed15M"],
        "South":   ["Florida",  "seed16S", "seed8S",  "seed9S",
                    "seed5S",   "seed12S", "seed4S",  "seed13S",
                    "seed6S",   "seed11S", "seed3S",  "seed14S",
                    "seed7S",   "seed10S", "seed2S",  "seed15S"],
    }


@pytest.fixture
def simple_wp_table(small_bracket):
    """Win probability table: 1-seeds ~85% vs 16-seeds, 50% mirror."""
    from src.model.win_probability import build_wp_table
    teams_data = {}
    seed_adj_em = {
        "Duke": 36.69, "Arizona": 35.54, "Michigan": 36.61, "Florida": 33.82,
    }
    for region_teams in small_bracket.values():
        for i, team in enumerate(region_teams):
            seed_pos = i  # 0=1-seed, 1=16-seed, etc.
            # Assign declining AdjEM by bracket position
            adj_em = seed_adj_em.get(team, 20.0 - seed_pos * 2.0)
            teams_data[team] = {"AdjEM": adj_em, "AdjT": 67.5}
    return build_wp_table(teams_data)
