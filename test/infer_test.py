"""Test module for trace_classifier/infer.py"""
from os import path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')
import os
from unittest import mock

import numpy as np

import findspark
findspark.init()

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from trace_classifier import infer
from .utils import FIXTURES_PATH, MODEL_PATH, TRACES_SCHEMA, is_equal_df


spark = SparkSession.builder \
    .config('spark.jars.packages', 'databricks:tensorframes:0.5.0-s_2.11') \
    .enableHiveSupport() \
    .appName("Generate routes test") \
    .getOrCreate()

traces_df = spark.read.json(
    path.join(FIXTURES_PATH, "traces.json"),
    schema=TRACES_SCHEMA
)


def test_infer_aggregated():
    actual_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=True)
    expected_df = spark.read.json(
        path.join(FIXTURES_PATH, "res_infer_aggregated.json")
    )
    probs_actual = actual_df.orderBy("test_id").select("probas").collect()
    probs_expected = expected_df.orderBy("test_id").select("probas").collect()
    assert all([ r for (ps_e, ps_a) in zip(probs_actual, probs_expected) for r in np.isclose(ps_e, ps_a).flatten() ])
    assert is_equal_df(
        expected_df.select("test_id", "pred_modality"),
        actual_df.select("test_id", "pred_modality"),
        sort_columns=["test_id"]
    )


def test_infer_unaggregated():
    actual_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=False)
    expected_df = spark.read.json(
        path.join(FIXTURES_PATH, "res_infer_unaggregated.json")
    )
    probs_actual = actual_df.orderBy("test_id", "phrase_pos").select("probas").collect()
    probs_expected = expected_df.orderBy("test_id", "phrase_pos").select("probas").collect()
    # Check probabilities are within floating point error
    assert all([ r for (ps_e, ps_a) in zip(probs_actual, probs_expected) for r in np.isclose(ps_e, ps_a).flatten() ])
    # Check predicted labels
    assert is_equal_df(
        expected_df.select("test_id", "phrase_pos", "pred_modality"),
        actual_df.select("test_id", "phrase_pos", "pred_modality"),
        sort_columns=["test_id", "phrase_pos"]
    )


def test_avg_probability():
    res_unaggregated_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=False)
    actual_df = infer.avg_probability(res_unaggregated_df, 'id', 'probas', 3)
    expected_df = spark.read.json(
        path.join(FIXTURES_PATH, "res_avg_probability.json")
    )
    probs_actual = actual_df.orderBy("id").select("sentence_probas").collect()
    probs_expected = expected_df.orderBy("id").select("sentence_probas").collect()
    assert all([ r for (ps_e, ps_a) in zip(probs_actual, probs_expected) for r in np.isclose(ps_e, ps_a).flatten() ])
    assert is_equal_df(
        expected_df.select("id", "sentence_pred_label"),
        actual_df.select("id", "sentence_pred_label"),
        sort_columns=["id"]
    )
