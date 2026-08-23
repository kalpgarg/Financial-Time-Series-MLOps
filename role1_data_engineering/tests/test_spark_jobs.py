"""Tests for role1 Spark cleaning logic.

Exercises the pure text-cleaning UDF used by clean_headlines. PySpark must be
importable to load the module (the function itself needs no SparkSession), so
the test skips cleanly where PySpark isn't installed.
"""

import pytest

pytest.importorskip("pyspark")

from role1_data_engineering.spark.clean_headlines import clean_text  # noqa: E402


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Multiple   Spaces  ") == "multiple spaces"


def test_clean_text_strips_html_tags():
    assert clean_text("<p>Tag</p>text") == "tagtext"
    assert clean_text("<b>Hello</b> World") == "hello world"


def test_clean_text_strips_html_entities():
    # "&amp;" matches the &<entity>; pattern and becomes a space, then collapses.
    assert clean_text("A&amp;B") == "a b"


def test_clean_text_handles_none():
    assert clean_text(None) is None
