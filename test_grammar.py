#!/usr/bin/env python3
"""Pytest tests to verify the SQL grammar can be loaded and used with outlines."""

import pytest
from pathlib import Path
from outlines.types import CFG


@pytest.fixture(scope="module")
def sql_grammar():
    """Fixture to load the SQL grammar from file."""
    grammar_path = Path("./sql.lark")
    assert grammar_path.exists(), "sql.lark file not found"
    return grammar_path.read_text()


@pytest.fixture(scope="module")
def outlines_cfg(sql_grammar):
    """Fixture to create an outlines CFG from the grammar."""
    return CFG(sql_grammar)


def test_grammar_file_exists():
    """Test that the grammar file exists."""
    grammar_path = Path("./sql.lark")
    assert grammar_path.exists(), "sql.lark file not found"


def test_grammar_loading(sql_grammar):
    """Test that the grammar file can be loaded and is not empty."""
    assert sql_grammar, "Grammar file is empty"
    assert len(sql_grammar) > 100, "Grammar file seems too short"
    assert "sql_script" in sql_grammar, "Grammar doesn't contain expected start rule"
    assert "select_stmt" in sql_grammar, "Grammar doesn't contain SELECT statement rule"


def test_outlines_cfg_creation(outlines_cfg):
    """Test that outlines can create a CFG from the grammar."""
    assert outlines_cfg is not None, "Failed to create outlines CFG"


def test_grammar_contains_key_rules(sql_grammar):
    """Test that the grammar contains all expected major SQL statement types."""
    expected_rules = [
        "select_stmt",
        "insert_stmt",
        "create_table_stmt",
        "with_clause",
        "where",
        "order_by_clause",
        "limit_clause",
    ]
    for rule in expected_rules:
        assert rule.lower() in sql_grammar.lower(), f"Grammar missing rule: {rule}"
