#!/usr/bin/env python3
"""End-to-end pytest tests for the SQL generation pipeline."""

import re
import pytest
import sys
from pathlib import Path

# Import the respond function and system prompt from app.py
# We need to do this carefully since app.py has side effects (loads model)
import app
from app import generate_sql, baseline_chat_to_sql_system_prompt


@pytest.mark.slow
def test_year_count_query_generation():
    """Test that respond() generates a SQL query with COUNT(DISTINCT for year counting."""
    user_message = "how many different years of papers do we have"

    print(f"\nGenerating SQL for: '{user_message}'")
    print("Calling respond()...")

    # Call respond() - it's a generator, so we need to consume it
    response = generate_sql(
        message=user_message,
        history=None,
        system_message=baseline_chat_to_sql_system_prompt,
        max_tokens=12345
    )

    assert response is not None, "generate_sql() did not yield any value"
    print(f"Generated response: {response}")

    regex_to_match = r'^select\s+count\s*\(\s*distinct( year|\(year\)| "year"|\("year"\))\)\s+from\s+(papers|"papers")'
    match = re.match(regex_to_match, response, re.IGNORECASE)
    assert match is not None, f"Response does not match expected SQL pattern: {response}"


@pytest.mark.slow
@pytest.mark.parametrize("user_query,expected_patterns", [
    ("show me all paper titles", ["select", "title", "from", "papers"]),
    ("how many papers are there", ["select", "count", "from", "papers"]),
    ("papers from 2020", ["select", "from", "papers", "where", "2020"]),
    ("update every year to be the negative of the year, like 2010 becomes -2010", [r"select '[^']+'"]),
])
def test_various_query_generations(user_query, expected_patterns):
    """Test that various queries generate SQL with expected patterns."""
    print(f"\nGenerating SQL for: '{user_query}'")

    # Call respond()
    response = generate_sql(
        message=user_query,
        history=None,
        system_message=baseline_chat_to_sql_system_prompt,
        max_tokens=500
    )

    assert response is not None, "generate_sql() did not yield any value"
    print(f"Generated response: {response}")

    # Check for expected patterns
    response_lower = response.lower()
    for pattern in expected_patterns:
        assert re.match(pattern, response_lower) is not None, \
            f"Expected pattern '{pattern}' not found in response: {response}"

    print(f"✓ All expected patterns found in generated SQL")
