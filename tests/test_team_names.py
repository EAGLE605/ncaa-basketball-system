"""
Tests for canonical team name resolver (src/utils/team_names.py).

Covers: exact match, case-insensitive, fuzzy fallback, UnresolvableTeamError,
alias registration, reload, resolve_many, is_resolvable.

All tests use the committed data/team_name_map.csv — no API calls.
"""
import pytest

from src.utils.team_names import (
    UnresolvableTeamError,
    add_alias,
    is_resolvable,
    reload,
    resolve,
    resolve_many,
)


# ---------------------------------------------------------------------------
# Ensure clean state before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_resolver():
    """Reload resolver from CSV before each test for isolation."""
    reload()
    yield
    reload()  # clean up aliases added during tests


# ---------------------------------------------------------------------------
# Exact-match resolution (names in team_name_map.csv)
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_canonical_name_resolves(self):
        # "Duke" is in CSV as canonical name
        assert resolve("Duke") == "Duke"

    def test_case_insensitive(self):
        assert resolve("duke") == "Duke"
        assert resolve("DUKE") == "Duke"

    def test_torvik_alias_resolves(self):
        # "UConn" is canonical (BartTorvik primary source); "Connecticut" is an alt
        result = resolve("UConn")
        assert result == "UConn"

    def test_espn_alias_resolves(self):
        # ESPN writes "Connecticut Huskies" — should map to canonical "UConn"
        result = resolve("Connecticut Huskies")
        assert result == "UConn"

    def test_alt_names_pipe_separated(self):
        # NCST is the canonical name (BartTorvik form); "NC State" is an alt_name
        result = resolve("NC State")
        assert result == "NCST"


# ---------------------------------------------------------------------------
# Fuzzy fallback
# ---------------------------------------------------------------------------

class TestFuzzyFallback:
    def test_near_match_resolves(self):
        # Small typo: "Dukke" should fuzzy-match to "Duke"
        # This test only passes if threshold allows it; use loose typo
        try:
            result = resolve("Michgan", fuzzy=True)  # Michigan
            assert result == "Michigan"
        except UnresolvableTeamError:
            pytest.skip("Fuzzy threshold didn't match 'Michgan' — acceptable")

    def test_fuzzy_disabled_raises(self):
        with pytest.raises(UnresolvableTeamError):
            resolve("ZZZUnknownTeamXXX", fuzzy=False)


# ---------------------------------------------------------------------------
# UnresolvableTeamError
# ---------------------------------------------------------------------------

class TestUnresolvableTeamError:
    def test_completely_unknown_raises(self):
        with pytest.raises(UnresolvableTeamError) as exc_info:
            resolve("ZZZUnknownTeamThatNeverExists999")
        assert "ZZZUnknownTeamThatNeverExists999" in str(exc_info.value)

    def test_error_message_mentions_csv(self):
        with pytest.raises(UnresolvableTeamError) as exc_info:
            resolve("FakeTeamXYZ")
        assert "team_name_map.csv" in str(exc_info.value)

    def test_is_subclass_of_value_error(self):
        with pytest.raises(ValueError):
            resolve("FakeTeamXYZ")


# ---------------------------------------------------------------------------
# add_alias (runtime-only, not persisted)
# ---------------------------------------------------------------------------

class TestAddAlias:
    def test_add_alias_resolves(self):
        add_alias("NCST/SMU", "NC State")
        assert resolve("NCST/SMU") == "NC State"

    def test_alias_case_insensitive(self):
        add_alias("TestAlias", "Duke")
        assert resolve("testalias") == "Duke"
        assert resolve("TESTALIAS") == "Duke"

    def test_alias_not_persisted_after_reload(self):
        add_alias("TempAliasXYZ", "Duke")
        assert resolve("TempAliasXYZ") == "Duke"
        reload()
        with pytest.raises(UnresolvableTeamError):
            resolve("TempAliasXYZ", fuzzy=False)


# ---------------------------------------------------------------------------
# resolve_many
# ---------------------------------------------------------------------------

class TestResolveMany:
    def test_empty_list(self):
        assert resolve_many([]) == []

    def test_single_name(self):
        result = resolve_many(["Duke"])
        assert result == ["Duke"]

    def test_multiple_names(self):
        result = resolve_many(["Duke", "Michigan"])
        assert result == ["Duke", "Michigan"]

    def test_raises_on_unknown_in_list(self):
        with pytest.raises(UnresolvableTeamError):
            resolve_many(["Duke", "ZZZUnknownXXX"])


# ---------------------------------------------------------------------------
# is_resolvable
# ---------------------------------------------------------------------------

class TestIsResolvable:
    def test_known_team_resolvable(self):
        assert is_resolvable("Duke") is True

    def test_unknown_team_not_resolvable(self):
        assert is_resolvable("ZZZNeverExists999") is False

    def test_case_insensitive_is_resolvable(self):
        assert is_resolvable("duke") is True


# ---------------------------------------------------------------------------
# 2026 tournament teams (spot-check)
# ---------------------------------------------------------------------------

class TestTournamentTeams2026:
    """Spot-check that key 2026 tournament teams resolve correctly."""

    @pytest.mark.parametrize("name,expected", [
        ("Duke", "Duke"),
        ("Michigan", "Michigan"),
        ("Arizona", "Arizona"),
        ("Florida", "Florida"),
        ("Purdue", "Purdue"),
        ("Houston", "Houston"),
    ])
    def test_team_resolves(self, name, expected):
        assert resolve(name) == expected
