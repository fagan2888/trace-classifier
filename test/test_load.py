"""Test module for trace_classifier/load.py"""
import json

from .utils import MODEL_PATH, METADATA_PATH
from trace_classifier import load

def test_load_model_metadata():
    expected = json.load(open(METADATA_PATH, "r"))
    actual = load.load_model_metadata(MODEL_PATH)
    assert expected == actual
