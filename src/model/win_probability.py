"""
Win probability model for NCAA Men's Basketball.

Ensemble architecture:
    P(win) = 0.65 * P_market + 0.35 * P_torvik   (when Pinnacle line available)
    P(win) = P_torvik                              (no market line)

BartTorvik formula:
    z = AdjEM_diff * tempo_factor * TOURNAMENT_MULTIPLIER / sigma
    P(win) = NormCDF(z)

where:
    AdjEM_diff      = team_a.AdjEM - team_b.AdjEM
    tempo_factor    = (tempo_a + tempo_b) / 200
    sigma           = 12.5 (3pt-heavy teams) or 11.0 (standard)
    tournament mult = 1.07 (variance compression in March)
"""

from scipy.stats import norm
from typing import Dict, Optional, Set

# Sigma values
SIGMA_DEFAULT    = 11.0
SIGMA_3PT_HEAVY  = 12.5

# Teams with higher 3pt variance (sigma=12.5)
TEAMS_3PT_HEAVY: Set[str] = {
    "Duke", "Michigan", "Kansas", "Alabama", "Purdue",
    "Gonzaga", "Arkansas", "Akron",
}

# Default tempo when not available (NCAA average ≈ 67-68 possessions per 40 min)
DEFAULT_TEMPO = 67.5

# Tournament pressure compresses variance slightly
TOURNAMENT_MULTIPLIER = 1.07

# Ensemble weights when market line is available
WEIGHT_MARKET  = 0.65
WEIGHT_TORVIK  = 0.35


def _sigma(team_a: str, team_b: str) -> float:
    """Return sigma for the matchup — wider if either team is 3pt-heavy."""
    if team_a in TEAMS_3PT_HEAVY or team_b in TEAMS_3PT_HEAVY:
        return SIGMA_3PT_HEAVY
    return SIGMA_DEFAULT


def torvik_win_prob(
    adjEM_a: float,
    adjEM_b: float,
    tempo_a: float = DEFAULT_TEMPO,
    tempo_b: float = DEFAULT_TEMPO,
    team_a: str = "",
    team_b: str = "",
) -> float:
    """
    BartTorvik-based win probability for team_a vs team_b.

    Args:
        adjEM_a:  AdjEM (AdjOE - AdjDE) for team A
        adjEM_b:  AdjEM for team B
        tempo_a:  Possessions per 40 min for team A
        tempo_b:  Possessions per 40 min for team B
        team_a:   Team name (for sigma lookup)
        team_b:   Team name (for sigma lookup)

    Returns:
        Float in (0, 1) — probability team A wins
    """
    sigma        = _sigma(team_a, team_b)
    tempo_factor = (tempo_a + tempo_b) / 200.0
    z            = (adjEM_a - adjEM_b) * tempo_factor * TOURNAMENT_MULTIPLIER / sigma
    return float(norm.cdf(z))


def market_win_prob(ml_home: int, ml_away: int, team_a_is_home: bool = True) -> float:
    """
    De-vigged win probability from American moneylines (multiplicative method).

    Args:
        ml_home:         Moneyline for the home team (e.g. -150 or +130)
        ml_away:         Moneyline for the away team
        team_a_is_home:  True if team_a is the home team

    Returns:
        Win probability for team_a
    """
    def raw(ml: int) -> float:
        return abs(ml) / (abs(ml) + 100) if ml < 0 else 100.0 / (ml + 100)

    rh, ra = raw(ml_home), raw(ml_away)
    total  = rh + ra
    p_home = rh / total
    return p_home if team_a_is_home else ra / total


def ensemble_win_prob(
    adjEM_a: float,
    adjEM_b: float,
    tempo_a: float = DEFAULT_TEMPO,
    tempo_b: float = DEFAULT_TEMPO,
    team_a: str = "",
    team_b: str = "",
    ml_team_a: Optional[int] = None,
    ml_team_b: Optional[int] = None,
) -> float:
    """
    Ensemble win probability combining BartTorvik model + Pinnacle market lines.

    Falls back to pure BartTorvik if moneylines are None.

    Args:
        adjEM_a/b:    AdjEM values
        tempo_a/b:    Possession rates
        team_a/b:     Team names
        ml_team_a:    American moneyline for team_a (None if no market line)
        ml_team_b:    American moneyline for team_b (None if no market line)

    Returns:
        Float in (0, 1) — probability team_a wins
    """
    p_torvik = torvik_win_prob(adjEM_a, adjEM_b, tempo_a, tempo_b, team_a, team_b)

    if ml_team_a is not None and ml_team_b is not None:
        p_market = market_win_prob(ml_team_a, ml_team_b, team_a_is_home=True)
        return WEIGHT_MARKET * p_market + WEIGHT_TORVIK * p_torvik

    return p_torvik


def build_wp_table(
    teams: Dict[str, Dict],
    market_lines: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Build a complete NxN win probability lookup table for a set of teams.

    Args:
        teams: {team_name: {"AdjEM": float, "AdjT": float}} — full tournament field
        market_lines: Optional {matchup_key: {"home": team, "home_ml": int, "away_ml": int}}
                      from OddsClient.extract_pinnacle_probs()

    Returns:
        {team_a: {team_b: p(a beats b)}} for all (a, b) pairs
    """
    team_names = list(teams.keys())
    table: Dict[str, Dict[str, float]] = {t: {} for t in team_names}

    for a in team_names:
        for b in team_names:
            if a == b:
                table[a][b] = 0.5
                continue

            ta_data = teams[a]
            tb_data = teams[b]
            em_a = ta_data.get("AdjEM") or ta_data.get("adjEM", 0.0)
            em_b = tb_data.get("AdjEM") or tb_data.get("adjEM", 0.0)
            t_a  = ta_data.get("AdjT") or DEFAULT_TEMPO
            t_b  = tb_data.get("AdjT") or DEFAULT_TEMPO

            # Clamp unrealistic AdjT values (BartTorvik sometimes returns rank, not tempo)
            if t_a < 55 or t_a > 80:
                t_a = DEFAULT_TEMPO
            if t_b < 55 or t_b > 80:
                t_b = DEFAULT_TEMPO

            # Look for market lines
            ml_a = ml_b = None
            if market_lines:
                key_ab = f"{a} vs {b}"
                key_ba = f"{b} vs {a}"
                if key_ab in market_lines:
                    entry = market_lines[key_ab]
                    if entry["home"] == a:
                        ml_a, ml_b = entry["home_ml"], entry["away_ml"]
                    else:
                        ml_b, ml_a = entry["home_ml"], entry["away_ml"]
                elif key_ba in market_lines:
                    entry = market_lines[key_ba]
                    if entry["home"] == b:
                        ml_b, ml_a = entry["home_ml"], entry["away_ml"]
                    else:
                        ml_a, ml_b = entry["home_ml"], entry["away_ml"]

            table[a][b] = ensemble_win_prob(em_a, em_b, t_a, t_b, a, b, ml_a, ml_b)

    return table
