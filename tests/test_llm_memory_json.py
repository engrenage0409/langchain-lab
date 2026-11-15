"""Qwen sometimes adds trailing text despite JSON instructions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_memory"))
from llm_memory import parse_qwen_json


def test_single_json_object():
    assert parse_qwen_json('{"summary":"ok","candidates":[]}')["summary"] == "ok"


def test_json_followed_by_extra_object():
    assert parse_qwen_json('{"summary":"first","candidates":[]}{"summary":"second"}')["summary"] == "first"


def test_fenced_json():
    assert parse_qwen_json('```json\n{"summary":"ok","candidates":[]}\n```')["summary"] == "ok"


def test_malformed_json_rejected():
    with pytest.raises(ValueError):
        parse_qwen_json('{"summary":')
