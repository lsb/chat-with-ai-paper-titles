#!/usr/bin/env python3
"""End-to-end pytest tests for the SQL generation pipeline."""

import re
import pytest
import sys
from pathlib import Path
import os
from flaky import flaky

# Import the respond function and system prompt from app.py
# We need to do this carefully since app.py has side effects (loads model)
import app
from app import generate_sql, generate_sql_expensive, baseline_chat_to_sql_system_prompt

year_regex_to_match = r'^\s*select\s+count\s*\(\s*distinct( year|\(year\)| "year"|\("year"\))\)\s*((as|AS) ([a-zA-Z_][a-zA-Z0-9_]*|"[a-zA-Z_][a-zA-Z0-9_]*"))?\s+from\s+(paper_authorship_records|"paper_authorship_records")'

@flaky(max_runs=3, min_passes=1)
@pytest.mark.slow
def test_year_count_query_generation():
    """Test that respond() generates a SQL query with COUNT(DISTINCT for year counting."""
    user_message = "how many different years of papers do we have"

    print(f"\nGenerating SQL for: '{user_message}'")
    print("Calling respond()...")

    response = generate_sql(
        message=user_message,
        history=None,
        system_message=baseline_chat_to_sql_system_prompt,
        max_tokens=500
    )

    assert response is not None, "generate_sql() did not yield any value"
    print(f"Generated response: {response}")

    match = re.match(year_regex_to_match, response, re.IGNORECASE)
    assert match is not None, f"Response does not match expected SQL pattern: {response}"

@flaky(max_runs=3, min_passes=1)
@pytest.mark.slow
def test_year_count_query_generation_expensive():
    """Test that respond() with expensive SQL generation generates a SQL query with COUNT(DISTINCT for year counting."""
    user_message = "how many different years of papers do we have"

    print(f"\nGenerating SQL for: '{user_message}'")
    print("Calling respond() with expensive generation...")

    response = generate_sql_expensive(
        message=user_message,
        history=None,
        system_message=baseline_chat_to_sql_system_prompt,
        max_tokens=12345,
    )

    assert response is not None, "generate_sql_expensive() did not yield any value"
    print(f"Generated response: {response}")

    match = re.match(year_regex_to_match, response, re.IGNORECASE)
    assert match is not None, f"Response does not match expected SQL pattern: {response}"

@flaky(max_runs=3, min_passes=1)
@pytest.mark.slow
@pytest.mark.parametrize("user_query,expected_patterns", [
    ("show me all paper titles", [r".*select.*", r".*title.*", r".*from.*", r".*paper_authorship.*"]),
    ("how many papers are there", [r".*select.*", r".*count.*", r".*from.*", r".*paper_authorships.*"]),
    ("papers from 2020", [r".*select.*", r".*from.*", r".*paper_authorships.*", r".*where.*", r".*2020.*"]),
    ("update every year to be the negative of the year, like 2010 becomes -2010", [r"^((?!update).)*$"]),  # should reject update
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
    response_expensive = generate_sql_expensive(
        message=user_query,
        history=None,
        system_message=baseline_chat_to_sql_system_prompt,
        max_tokens=12345,
    )

    assert response is not None, "generate_sql() did not yield any value"
    assert response_expensive is not None, "generate_sql_expensive() did not yield any value"
    print(f"Generated response: {response}")

    # Check for expected patterns
    response_lower = response.lower()
    response_expensive_lower = response_expensive.lower()
    for pattern in expected_patterns:
        assert re.match(pattern, response_lower) is not None, f"pattern '{pattern}' not in response {response}"
        assert re.match(pattern, response_expensive_lower) is not None, f"pattern '{pattern}' not in expensive response {response_expensive}"

    print(f"✓ All expected patterns found in generated SQL")
