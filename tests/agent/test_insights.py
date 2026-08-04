"""Tests for agent/insights.py — InsightsEngine analytics and reporting."""

import sqlite3
import time
import pytest

import hermes_state
from hermes_state import SessionDB
from agent.insights import (
    InsightsEngine,
    _estimate_cost,
    _bar_chart,
    open_insights_db,
)
from agent.usage_pricing import (
    format_duration_compact as _format_duration,
    has_known_pricing as _has_known_pricing,
)


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_insights.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def populated_db(db):
    """Create a DB with realistic session data for insights testing."""
    now = time.time()
    day = 86400

    # Session 1: CLI, claude-sonnet, ended, 2 days ago
    db.create_session(
        session_id="s1", source="cli",
        model="anthropic/claude-sonnet-4-20250514", user_id="user1",
    )
    # Backdate the started_at
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 2 * day,))
    db.end_session("s1", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's1'", (now - 2 * day + 3600,))
    db.update_token_counts("s1", input_tokens=50000, output_tokens=15000)
    db.append_message("s1", role="user", content="Hello, help me fix a bug")
    db.append_message("s1", role="assistant", content="Sure, let me look into that.")
    db.append_message("s1", role="assistant", content="Let me search the files.",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s1", role="tool", content="Found 3 matches", tool_name="search_files")
    db.append_message("s1", role="assistant", content="Let me read the file.",
                      tool_calls=[{"function": {"name": "read_file"}}])
    db.append_message("s1", role="tool", content="file contents...", tool_name="read_file")
    db.append_message("s1", role="assistant", content="I found the bug. Let me fix it.",
                      tool_calls=[{"function": {"name": "patch"}}])
    db.append_message("s1", role="tool", content="patched successfully", tool_name="patch")
    db.append_message(
        "s1",
        role="assistant",
        content="Let me load the PR workflow skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}}],
    )
    db.append_message("s1", role="user", content="Thanks!")
    db.append_message("s1", role="assistant", content="You're welcome!")

    # Session 2: Telegram, gpt-4o, ended, 5 days ago
    db.create_session(
        session_id="s2", source="telegram",
        model="gpt-4o", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's2'", (now - 5 * day,))
    db.end_session("s2", end_reason="timeout")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's2'", (now - 5 * day + 1800,))
    db.update_token_counts("s2", input_tokens=20000, output_tokens=8000)
    db.append_message("s2", role="user", content="Search the web for something")
    db.append_message("s2", role="assistant", content="Searching...",
                      tool_calls=[{"function": {"name": "web_search"}}])
    db.append_message("s2", role="tool", content="results...", tool_name="web_search")
    db.append_message("s2", role="assistant", content="Here's what I found")

    # Session 3: CLI, deepseek-chat, ended, 10 days ago
    db.create_session(
        session_id="s3", source="cli",
        model="deepseek-chat", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's3'", (now - 10 * day,))
    db.end_session("s3", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's3'", (now - 10 * day + 7200,))
    db.update_token_counts("s3", input_tokens=100000, output_tokens=40000)
    db.append_message("s3", role="user", content="Run this terminal command")
    db.append_message("s3", role="assistant", content="Running...",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="output...", tool_name="terminal")
    db.append_message("s3", role="assistant", content="Let me run another",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="more output...", tool_name="terminal")
    db.append_message("s3", role="assistant", content="And search files",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s3", role="tool", content="found stuff", tool_name="search_files")
    db.append_message(
        "s3",
        role="assistant",
        content="Load the debugging skill.",
        tool_calls=[{"function": {"name": "skill_view", "arguments": '{"name":"systematic-debugging"}'}}],
    )

    # Session 4: Discord, same model as s1, ended, 1 day ago
    db.create_session(
        session_id="s4", source="discord",
        model="anthropic/claude-sonnet-4-20250514", user_id="user2",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's4'", (now - 1 * day,))
    db.end_session("s4", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's4'", (now - 1 * day + 900,))
    db.update_token_counts("s4", input_tokens=10000, output_tokens=5000)
    db.append_message("s4", role="user", content="Quick question")
    db.append_message("s4", role="assistant", content="Sure, go ahead")
    db.append_message(
        "s4",
        role="assistant",
        content="Load and update GitHub skills.",
        tool_calls=[
            {"function": {"name": "skill_view", "arguments": '{"name":"github-pr-workflow"}'}},
            {"function": {"name": "skill_manage", "arguments": '{"name":"github-code-review"}'}},
        ],
    )

    # Session 5: Old session, 45 days ago (should be excluded from 30-day window)
    db.create_session(
        session_id="s_old", source="cli",
        model="gpt-4o-mini", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's_old'", (now - 45 * day,))
    db.end_session("s_old", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's_old'", (now - 45 * day + 600,))
    db.update_token_counts("s_old", input_tokens=5000, output_tokens=2000)
    db.append_message("s_old", role="user", content="old message")
    db.append_message("s_old", role="assistant", content="old reply")

    db._conn.commit()
    return db


class TestHasKnownPricing:
    def test_known_commercial_model(self):
        assert _has_known_pricing("gpt-4o", provider="openai") is True
        assert _has_known_pricing("anthropic/claude-sonnet-4-20250514") is True
        assert _has_known_pricing("gpt-4.1", provider="openai") is True

    def test_unknown_custom_model(self):
        assert _has_known_pricing("FP16_Hermes_4.5") is False
        assert _has_known_pricing("my-custom-model") is False
        assert _has_known_pricing("glm-5") is False
        assert _has_known_pricing("") is False
        assert _has_known_pricing(None) is False

    def test_heuristic_matched_models_are_not_considered_known(self):
        assert _has_known_pricing("some-opus-model") is False
        assert _has_known_pricing("future-sonnet-v2") is False


class TestEstimateCost:
    def test_basic_cost(self):
        cost, status = _estimate_cost(
            "anthropic/claude-sonnet-4-20250514",
            1_000_000,
            1_000_000,
            provider="anthropic",
        )
        assert status == "estimated"
        assert cost == pytest.approx(18.0, abs=0.01)

    def test_zero_tokens(self):
        cost, status = _estimate_cost("gpt-4o", 0, 0, provider="openai")
        assert status == "estimated"
        assert cost == 0.0

    def test_cache_aware_usage(self):
        cost, status = _estimate_cost(
            "anthropic/claude-sonnet-4-20250514",
            1000,
            500,
            cache_read_tokens=2000,
            cache_write_tokens=400,
            provider="anthropic",
        )
        assert status == "estimated"
        expected = (1000 * 3.0 + 500 * 15.0 + 2000 * 0.30 + 400 * 3.75) / 1_000_000
        assert cost == pytest.approx(expected, abs=0.0001)


# =========================================================================
# Format helpers
# =========================================================================

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(300) == "5m"

    def test_hours_with_minutes(self):
        result = _format_duration(5400)  # 1.5 hours
        assert result == "1h 30m"

    def test_exact_hours(self):
        assert _format_duration(7200) == "2h"

    def test_days(self):
        result = _format_duration(172800)  # 2 days
        assert result == "2.0d"


class TestBarChart:
    def test_basic_bars(self):
        bars = _bar_chart([10, 5, 0, 20], max_width=10)
        assert len(bars) == 4
        assert len(bars[3]) == 10  # max value gets full width
        assert len(bars[0]) == 5   # half of max
        assert bars[2] == ""       # zero gets empty

    def test_empty_values(self):
        bars = _bar_chart([], max_width=10)
        assert bars == []

    def test_all_zeros(self):
        bars = _bar_chart([0, 0, 0], max_width=10)
        assert all(b == "" for b in bars)

    def test_single_value(self):
        bars = _bar_chart([5], max_width=10)
        assert len(bars) == 1
        assert len(bars[0]) == 10


# =========================================================================
# InsightsEngine — empty DB
# =========================================================================

class TestInsightsEmpty:
    def test_empty_db_returns_empty_report(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is True
        assert report["overview"] == {}

    def test_empty_db_terminal_format(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)
        assert "No sessions found" in text

    def test_empty_db_gateway_format(self, db):
        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)
        assert "No sessions found" in text


# =========================================================================
# InsightsEngine — populated DB
# =========================================================================

class TestInsightsPopulated:
    def test_generate_returns_all_sections(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)

        assert report["empty"] is False
        assert "overview" in report
        assert "models" in report
        assert "platforms" in report
        assert "tools" in report
        assert "activity" in report
        assert "top_sessions" in report

    def test_overview_session_count(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        # s1, s2, s3, s4 are within 30 days; s_old is 45 days ago
        assert overview["total_sessions"] == 4

    def test_overview_token_totals(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        expected_input = 50000 + 20000 + 100000 + 10000
        expected_output = 15000 + 8000 + 40000 + 5000
        assert overview["total_input_tokens"] == expected_input
        assert overview["total_output_tokens"] == expected_output
        assert overview["total_tokens"] == expected_input + expected_output

    def test_overview_cost_positive(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        assert report["overview"]["estimated_cost"] > 0

    def test_overview_duration_stats(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        overview = report["overview"]

        # All 4 sessions have durations
        assert overview["total_hours"] > 0
        assert overview["avg_session_duration"] > 0

    def test_model_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        models = report["models"]

        # Should have 3 distinct models (claude-sonnet x2, gpt-4o, deepseek-chat)
        model_names = [m["model"] for m in models]
        assert "claude-sonnet-4-20250514" in model_names
        assert "gpt-4o" in model_names
        assert "deepseek-chat" in model_names

        # Claude-sonnet has 2 sessions (s1 + s4)
        claude = next(m for m in models if "claude-sonnet" in m["model"])
        assert claude["sessions"] == 2

    def test_model_breakdown_splits_mid_session_switch(self, db):
        """A session that switches models mid-flight is split across both
        models in the breakdown, not dumped on the initial model (#51607).
        """
        now = time.time()
        db.create_session(session_id="sw", source="cli",
                          model="deepseek/deepseek-v4-pro")
        # 40k tokens on deepseek, then switch and 50k on opus.
        db.update_token_counts("sw", input_tokens=40000, output_tokens=8000,
                               model="deepseek/deepseek-v4-pro",
                               billing_provider="deepseek", api_call_count=2)
        db.update_session_model("sw", "anthropic/claude-opus-4.8")
        db.update_token_counts("sw", input_tokens=50000, output_tokens=4000,
                               model="anthropic/claude-opus-4.8",
                               billing_provider="openrouter", api_call_count=3)
        db._conn.commit()

        report = InsightsEngine(db).generate(days=30)
        models = {m["model"]: m for m in report["models"]}
        assert "deepseek-v4-pro" in models
        assert "claude-opus-4.8" in models
        # Tokens attributed to the model that actually incurred them.
        assert models["deepseek-v4-pro"]["input_tokens"] == 40000
        assert models["claude-opus-4.8"]["input_tokens"] == 50000
        assert models["claude-opus-4.8"]["api_calls"] == 3
        # The summary row's single model would have hidden one of these.
        assert models["deepseek-v4-pro"]["total_tokens"] == 48000
        assert models["claude-opus-4.8"]["total_tokens"] == 54000

    def test_partial_per_model_rows_preserve_session_totals(self, db):
        """A partial rolling-upgrade row must not hide aggregate residuals."""
        db.create_session(session_id="partial", source="cli", model="gpt-4o")
        db.update_token_counts(
            "partial", input_tokens=100, output_tokens=20,
            model="gpt-4o", billing_provider="openai", api_call_count=1,
        )
        db.update_token_counts(
            "partial", input_tokens=1000, output_tokens=200,
            model="gpt-4o", billing_provider="openai", api_call_count=10,
            absolute=True,
        )

        report = InsightsEngine(db).generate(days=30)
        model = next(m for m in report["models"] if m["model"] == "gpt-4o")
        assert model["input_tokens"] == 1000
        assert model["output_tokens"] == 200
        assert model["api_calls"] == 10
        assert sum(m["total_tokens"] for m in report["models"]) == \
            report["overview"]["total_tokens"]

    def test_overview_cost_matches_per_model_stored_cost(self, db):
        db.create_session(session_id="cost", source="cli", model="model-a")
        db.update_token_counts(
            "cost", input_tokens=10, model="model-a", billing_provider="custom",
            estimated_cost_usd=1.25, actual_cost_usd=1.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )
        db.update_session_model("cost", "model-b")
        db.update_session_billing_route("cost", provider="custom-b", base_url=None)
        db.update_token_counts(
            "cost", input_tokens=20, model="model-b", billing_provider="custom-b",
            estimated_cost_usd=2.5, actual_cost_usd=2.0,
            cost_status="estimated", cost_source="provider", api_call_count=1,
        )

        report = InsightsEngine(db).generate(days=30)
        assert sum(m["cost"] for m in report["models"]) == pytest.approx(3.75)
        assert report["overview"]["estimated_cost"] == pytest.approx(3.75)
        assert report["overview"]["actual_cost"] == pytest.approx(3.0)

    def test_platform_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        platforms = report["platforms"]

        platform_names = [p["platform"] for p in platforms]
        assert "cli" in platform_names
        assert "telegram" in platform_names
        assert "discord" in platform_names

        cli = next(p for p in platforms if p["platform"] == "cli")
        assert cli["sessions"] == 2  # s1 + s3

    def test_tool_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        tools = report["tools"]

        tool_names = [t["tool"] for t in tools]
        assert "terminal" in tool_names
        assert "search_files" in tool_names
        assert "read_file" in tool_names
        assert "patch" in tool_names
        assert "web_search" in tool_names

        # terminal was used 2x in s3
        terminal = next(t for t in tools if t["tool"] == "terminal")
        assert terminal["count"] == 2

        # Percentages should sum to ~100%
        total_pct = sum(t["percentage"] for t in tools)
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_skill_breakdown(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        skills = report["skills"]

        assert skills["summary"]["distinct_skills_used"] == 3
        assert skills["summary"]["total_skill_loads"] == 3
        assert skills["summary"]["total_skill_edits"] == 1
        assert skills["summary"]["total_skill_actions"] == 4

        top_skill = skills["top_skills"][0]
        assert top_skill["skill"] == "github-pr-workflow"
        assert top_skill["view_count"] == 2
        assert top_skill["manage_count"] == 0
        assert top_skill["total_count"] == 2
        assert top_skill["last_used_at"] is not None

    def test_skill_breakdown_respects_days_filter(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=3)
        skills = report["skills"]

        assert skills["summary"]["distinct_skills_used"] == 2
        assert skills["summary"]["total_skill_loads"] == 2
        assert skills["summary"]["total_skill_edits"] == 1

        skill_names = [s["skill"] for s in skills["top_skills"]]
        assert "systematic-debugging" not in skill_names

    def test_activity_patterns(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        activity = report["activity"]

        assert len(activity["by_day"]) == 7
        assert len(activity["by_hour"]) == 24
        assert activity["active_days"] >= 1
        assert activity["busiest_day"] is not None
        assert activity["busiest_hour"] is not None

    def test_top_sessions(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        top = report["top_sessions"]

        labels = [t["label"] for t in top]
        assert "Longest session" in labels
        assert "Most messages" in labels
        assert "Most tokens" in labels
        assert "Most tool calls" in labels

    def test_source_filter_cli(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="cli")

        assert report["overview"]["total_sessions"] == 2  # s1, s3

    def test_source_filter_telegram(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="telegram")

        assert report["overview"]["total_sessions"] == 1  # s2

    def test_source_filter_nonexistent(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30, source="slack")

        assert report["empty"] is True

    def test_days_filter_short(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=3)

        # Only s1 (2 days ago) and s4 (1 day ago) should be included
        assert report["overview"]["total_sessions"] == 2

    def test_days_filter_long(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=60)

        # All 5 sessions should be included
        assert report["overview"]["total_sessions"] == 5


# =========================================================================
# Formatting
# =========================================================================

class TestTerminalFormatting:
    def test_terminal_format_has_sections(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "Hermes Insights" in text
        assert "Overview" in text
        assert "Models Used" in text
        assert "Top Tools" in text
        assert "Top Skills" in text
        assert "Activity Patterns" in text
        assert "Notable Sessions" in text

    def test_terminal_format_shows_tokens(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "Input tokens" in text
        assert "Output tokens" in text
        # Cost and cache metrics are intentionally hidden (pricing was unreliable).
        assert "Est. cost" not in text
        assert "Cache read" not in text
        assert "Cache write" not in text

    def test_terminal_format_shows_platforms(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        # Multi-platform, so Platforms section should show
        assert "Platforms" in text
        assert "cli" in text
        assert "telegram" in text

    def test_terminal_format_shows_bar_chart(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "█" in text  # Bar chart characters

    def test_terminal_format_hides_cost_for_custom_models(self, db):
        """Cost display is hidden entirely — custom models no longer show 'N/A' either."""
        db.create_session(session_id="s1", source="cli", model="my-custom-model")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        text = engine.format_terminal(report)

        assert "N/A" not in text
        assert "custom/self-hosted" not in text
        assert "Cost" not in text


class TestGatewayFormatting:
    def test_gateway_format_is_shorter(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        terminal_text = engine.format_terminal(report)
        gateway_text = engine.format_gateway(report)

        assert len(gateway_text) < len(terminal_text)

    def test_gateway_format_has_bold(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "**" in text  # Markdown bold

    def test_gateway_format_hides_cost(self, populated_db):
        """Gateway format omits dollar figures and internal cache details."""
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "$" not in text
        assert "cache" not in text.lower()

    def test_gateway_format_shows_models(self, populated_db):
        engine = InsightsEngine(populated_db)
        report = engine.generate(days=30)
        text = engine.format_gateway(report)

        assert "Models" in text
        assert "sessions" in text


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_session_with_no_tokens(self, db):
        """Sessions with zero tokens should not crash."""
        db.create_session(session_id="s1", source="cli", model="test-model")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is False
        assert report["overview"]["total_tokens"] == 0
        assert report["overview"]["estimated_cost"] == 0.0

    def test_session_with_no_end_time(self, db):
        """Active (non-ended) sessions should be included but duration = 0."""
        db.create_session(session_id="s1", source="cli", model="test-model")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        # Session included
        assert report["overview"]["total_sessions"] == 1
        assert report["overview"]["total_tokens"] == 1500
        # But no duration stats (session not ended)
        assert report["overview"]["total_hours"] == 0

    def test_session_with_no_model(self, db):
        """Sessions with NULL model should not crash."""
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=1000, output_tokens=500)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["empty"] is False

        models = report["models"]
        assert len(models) == 1
        assert models[0]["model"] == "unknown"
        assert models[0]["has_pricing"] is False

    def test_custom_model_shows_zero_cost(self, db):
        """Custom/self-hosted models should show $0 cost, not fake estimates."""
        db.create_session(session_id="s1", source="cli", model="FP16_Hermes_4.5")
        db.update_token_counts("s1", input_tokens=100000, output_tokens=50000)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["overview"]["estimated_cost"] == 0.0
        assert "FP16_Hermes_4.5" in report["overview"]["models_without_pricing"]

        models = report["models"]
        custom = next(m for m in models if m["model"] == "FP16_Hermes_4.5")
        assert custom["cost"] == 0.0
        assert custom["has_pricing"] is False

    def test_tool_usage_from_tool_calls_json(self, db):
        """Tool usage should be extracted from tool_calls JSON when tool_name is NULL."""
        db.create_session(session_id="s1", source="cli", model="test")
        # Assistant message with tool_calls (this is what CLI produces)
        db.append_message("s1", role="assistant", content="Let me search",
                          tool_calls=[{"id": "call_1", "type": "function",
                                       "function": {"name": "search_files", "arguments": "{}"}}])
        # Tool response WITHOUT tool_name (this is the CLI bug)
        db.append_message("s1", role="tool", content="found results",
                          tool_call_id="call_1")
        db.append_message("s1", role="assistant", content="Now reading",
                          tool_calls=[{"id": "call_2", "type": "function",
                                       "function": {"name": "read_file", "arguments": "{}"}}])
        db.append_message("s1", role="tool", content="file content",
                          tool_call_id="call_2")
        db.append_message("s1", role="assistant", content="And searching again",
                          tool_calls=[{"id": "call_3", "type": "function",
                                       "function": {"name": "search_files", "arguments": "{}"}}])
        db.append_message("s1", role="tool", content="more results",
                          tool_call_id="call_3")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        tools = report["tools"]

        # Should find tools from tool_calls JSON even though tool_name is NULL
        tool_names = [t["tool"] for t in tools]
        assert "search_files" in tool_names
        assert "read_file" in tool_names

        # search_files was called twice
        sf = next(t for t in tools if t["tool"] == "search_files")
        assert sf["count"] == 2

    def test_overview_pricing_sets_are_lists(self, db):
        """models_with/without_pricing should be JSON-serializable lists."""
        import json as _json
        db.create_session(session_id="s1", source="cli", model="gpt-4o")
        db.create_session(session_id="s2", source="cli", model="my-custom")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        overview = report["overview"]

        assert isinstance(overview["models_with_pricing"], list)
        assert isinstance(overview["models_without_pricing"], list)
        # Should be JSON-serializable
        _json.dumps(report["overview"])  # would raise if sets present

    def test_mixed_commercial_and_custom_models(self, db):
        """Mix of commercial and custom models: only commercial ones get costs."""
        db.create_session(session_id="s1", source="cli", model="anthropic/claude-sonnet-4-20250514")
        db.update_token_counts(
            "s1",
            input_tokens=10000,
            output_tokens=5000,
            billing_provider="anthropic",
        )
        db.create_session(session_id="s2", source="cli", model="my-local-llama")
        db.update_token_counts("s2", input_tokens=10000, output_tokens=5000)
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)

        # Cost should only come from gpt-4o, not from the custom model
        overview = report["overview"]
        assert overview["estimated_cost"] > 0
        assert "claude-sonnet-4-20250514" in overview["models_with_pricing"]  # list now, not set
        assert "my-local-llama" in overview["models_without_pricing"]

        # Verify individual model entries
        claude = next(m for m in report["models"] if m["model"] == "claude-sonnet-4-20250514")
        assert claude["has_pricing"] is True
        assert claude["cost"] > 0

        llama = next(m for m in report["models"] if m["model"] == "my-local-llama")
        assert llama["has_pricing"] is False
        assert llama["cost"] == 0.0

    def test_single_session_streak(self, db):
        """Single session should have streak of 0 or 1."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["activity"]["max_streak"] <= 1

    def test_no_tool_calls(self, db):
        """Sessions with no tool calls should produce empty tools list."""
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", role="user", content="hello")
        db.append_message("s1", role="assistant", content="hi there")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert report["tools"] == []

    def test_only_one_platform(self, db):
        """Single-platform usage should still work."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=30)
        assert len(report["platforms"]) == 1
        assert report["platforms"][0]["platform"] == "cli"

        # Terminal format should NOT show platform section for single platform
        text = engine.format_terminal(report)
        # (it still shows platforms section if there's only cli and nothing else)
        # Actually the condition is > 1 platforms OR non-cli, so single cli won't show

    def test_large_days_value(self, db):
        """Very large days value should not crash."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=365)
        assert report["empty"] is False

    def test_zero_days(self, db):
        """Zero days should return empty (nothing is in the future)."""
        db.create_session(session_id="s1", source="cli", model="test")
        db._conn.commit()

        engine = InsightsEngine(db)
        report = engine.generate(days=0)
        # Depending on timing, might catch the session if created <1s ago
        # Just verify it doesn't crash
        assert "empty" in report


# =========================================================================
# Reporting must not become a writer
# =========================================================================

class TestInsightsDoesNotTakeTheWriteLock:
    """``hermes insights`` runs SELECTs and must open ``state.db`` read-only.

    Context (vpsclone, 2026-08-03 23:03:59): ``hermes insights --days 30`` was
    the last tool launched before every writer on a 14.5 GB ``state.db`` began
    failing with ``database is locked``; it ran until the terminal timed out at
    23:05:59.  The exact lock-holding statement was never proven, but a
    reporting command has no business opening a writable handle at all: a
    writable ``SessionDB()`` runs the schema-init pass (which writes, with a
    deliberately widened busy handler) and checkpoints the WAL on ``close()``.
    Removing that removes a whole class of suspect without changing any report.
    """

    def test_existing_database_is_opened_read_only(self, tmp_path, monkeypatch):
        db_path = tmp_path / "state.db"
        SessionDB(db_path=db_path).close()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        db = open_insights_db()
        try:
            assert db.read_only is True
            # Not merely a flag: the handle genuinely cannot write, so it can
            # neither take nor wait for the single write lock.
            with pytest.raises(sqlite3.OperationalError):
                db._conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES ('probe', '1')"
                )
            # And it still reads, which is the whole job.
            assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        finally:
            db.close()

    def test_report_still_generates_over_the_read_only_handle(
        self, tmp_path, monkeypatch
    ):
        """End to end: a real report over a real read-only attach."""
        db_path = tmp_path / "state.db"
        seed = SessionDB(db_path=db_path)
        seed.create_session(session_id="s1", source="cli", model="test-model")
        seed.update_token_counts("s1", input_tokens=100, output_tokens=50)
        seed.close()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        db = open_insights_db()
        try:
            report = InsightsEngine(db).generate(days=30)
            assert report["empty"] is False
            assert report["overview"]["total_tokens"] == 150
        finally:
            db.close()

    def test_first_run_reports_without_creating_a_database(
        self, tmp_path, monkeypatch
    ):
        """Read-only even on the FIRST run: no file, no writes, still a report.

        The earlier shape of this change opened a writable ``SessionDB()`` when
        the file was missing, guarded by ``db_path.exists()``.  Two problems:
        the guard is a TOCTOU window (another process can create the file in
        between, at which point insights becomes exactly the writer this change
        removes), and creating + migrating a database is a strange thing for a
        reporting command to do.  An in-memory empty stand-in gives the same
        user-visible result with no writer and no file.
        """
        db_path = tmp_path / "fresh" / "state.db"
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        db = open_insights_db()
        try:
            assert db.read_only is True
            assert not db_path.exists(), "reporting must not create state.db"
            report = InsightsEngine(db).generate(days=30)
            assert report["empty"] is True
        finally:
            db.close()
        # Not even the directory: reporting performs no filesystem writes.
        assert not db_path.parent.exists()

    def test_unopenable_existing_path_is_an_error_not_a_writable_open(
        self, tmp_path, monkeypatch
    ):
        """A path that exists but cannot be opened read-only must NOT be upgraded.

        Silently falling back to a writable open would reintroduce the writer;
        silently reporting "no data" would lie about a database that has data.
        """
        db_path = tmp_path / "state.db"
        db_path.mkdir()  # exists, and can never be opened as a database
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        with pytest.raises(RuntimeError, match="never opens state.db for writing"):
            open_insights_db()
        # Untouched: no schema init, no migration, no quarantine, no new file.
        assert db_path.is_dir()
        assert list(db_path.iterdir()) == []

    def test_cold_wal_database_opens_read_only_without_a_live_shm(
        self, tmp_path, monkeypatch
    ):
        """The common CLI case: nothing else has the database open.

        A cleanly closed WAL database has no ``-wal``/``-shm`` sidecars, so a
        read-only open has to establish the WAL index itself.  This is the
        single most likely way the read-only switch could break ``hermes
        insights`` for everyone, so it is asserted rather than assumed.
        """
        db_path = tmp_path / "state.db"
        seed = SessionDB(db_path=db_path)
        seed.create_session(session_id="cold", source="cli", model="m")
        seed.append_message(session_id="cold", role="user", content="hi")
        seed.close()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
        # Cold start premise: a clean close removes the WAL sidecars, so the
        # read-only open below has to establish the WAL index itself. If a
        # platform keeps them, the premise does not hold and the test would be
        # vacuous rather than wrong.
        if (tmp_path / "state.db-shm").exists():
            pytest.skip("platform kept the -shm sidecar after close")

        db = open_insights_db()
        try:
            assert db.read_only is True
            report = InsightsEngine(db).generate(days=30)
            assert report["empty"] is False
            assert report["overview"]["total_sessions"] == 1
        finally:
            db.close()

    @pytest.mark.parametrize(
        "dirname", ["has?question", "has#hash", "has%20literal", "has space"]
    )
    def test_uri_special_characters_in_the_path_are_escaped(
        self, tmp_path, monkeypatch, dirname
    ):
        """``file:{path}?mode=ro`` is a URI, so the path must be escaped.

        Unescaped, a ``HERMES_HOME`` containing ``?``, ``#`` or ``%NN`` either
        opens a DIFFERENT database or fails with an opaque "unable to open
        database file" — and this change makes every ``hermes insights``
        invocation depend on that parsing.
        """
        home = tmp_path / dirname
        home.mkdir()
        db_path = home / "state.db"
        seed = SessionDB(db_path=db_path)
        seed.create_session(session_id="s-special", source="cli", model="m")
        seed.close()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        db = open_insights_db()
        try:
            assert db.read_only is True
            # The RIGHT database: the session we seeded is visible.
            rows = db._conn.execute("SELECT id FROM sessions").fetchall()
            assert [r[0] for r in rows] == ["s-special"]
        finally:
            db.close()

    def test_report_degrades_on_a_pre_migration_schema(self, tmp_path, monkeypatch):
        """An older database must still produce a report, not an error.

        A read-only open deliberately does not run migrations, so the first
        command after an update — plausibly ``hermes insights`` — can meet a
        ``sessions`` table without the newer cost/billing columns.  Before the
        degradation guard that raised ``no such column`` straight into "Error
        generating insights".
        """
        db_path = tmp_path / "state.db"
        # Build a deliberately OLD sessions table: identity + a couple of
        # counters, none of the columns later migrations added.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, "
                "model TEXT, started_at REAL, ended_at REAL, "
                "message_count INTEGER, input_tokens INTEGER, "
                "output_tokens INTEGER)"
            )
            conn.execute(
                "INSERT INTO sessions (id, source, model, started_at, "
                "message_count, input_tokens, output_tokens) "
                "VALUES ('old', 'cli', 'legacy-model', ?, 4, 100, 50)",
                (time.time(),),
            )
            conn.execute(
                "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
                "role TEXT, content TEXT, timestamp REAL, tool_name TEXT, "
                "tool_calls TEXT)"
            )
            conn.commit()
        finally:
            conn.close()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

        db = open_insights_db()
        try:
            engine = InsightsEngine(db)
            report = engine.generate(days=30)
            assert report["empty"] is False
            assert report["degraded"] is True
            # The columns that exist are still summarised correctly...
            assert report["overview"]["total_sessions"] == 1
            assert report["overview"]["total_tokens"] == 150
            # ...and the missing ones are named for the operator.
            assert "cost_status" in report["missing_session_columns"]
            # Formatting must not blow up on the reduced row either.
            assert "Sessions" in engine.format_terminal(report)
        finally:
            db.close()
