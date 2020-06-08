"""Test module for trace_classifier/infer.py"""
import logging
from os import path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ [%(levelname)s][%(name)s] %(message)s"
)

from pyspark.sql import SparkSession

from trace_classifier import infer
from .utils import (
    FIXTURES_PATH,
    MODEL_PATH,
    TRACES_SCHEMA,
    is_equal_df,
    assert_are_close,
)


spark = (
    SparkSession.builder.config(
        "spark.jars.packages", "databricks:tensorframes:0.5.0-s_2.11"
    )
    .enableHiveSupport()
    .appName("Generate routes test")
    .getOrCreate()
)

traces_df = spark.read.json(
    path.join(FIXTURES_PATH, "traces.json"), schema=TRACES_SCHEMA
)
short_traces_df = spark.read.json(
    path.join(FIXTURES_PATH, "short_traces.json"), schema=TRACES_SCHEMA
)


def test_infer_aggregated():
    actual_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=True)
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_infer_aggregated.json"))
    probs_actual = actual_df.orderBy("test_id").select("probas").collect()
    probs_expected = expected_df.orderBy("test_id").select("probas").collect()
    assert_are_close(probs_actual, probs_expected)
    assert is_equal_df(
        expected_df.select("test_id", "pred_modality"),
        actual_df.select("test_id", "pred_modality"),
        sort_columns=["test_id"],
    )

    # Ensure short traces aren't dropped
    # NOTE: the sample model used for testing has a word length of 2
    actual_df_short = infer.infer(
        short_traces_df, model_file=MODEL_PATH, aggregate=True
    )
    expected_df_short = spark.read.json(
        path.join(FIXTURES_PATH, "res_infer_short.json")
    )
    probs_actual_short = actual_df_short.orderBy("test_id").select("probas").collect()
    probs_expected_short = (
        expected_df_short.orderBy("test_id").select("probas").collect()
    )
    assert_are_close(probs_actual_short, probs_expected_short)
    assert actual_df_short.count() == 3
    assert is_equal_df(
        expected_df_short.select("test_id", "pred_modality"),
        actual_df_short.select("test_id", "pred_modality"),
        sort_columns=["test_id"],
    )


def test_infer_unaggregated():
    actual_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=False)
    expected_df = spark.read.json(
        path.join(FIXTURES_PATH, "res_infer_unaggregated.json")
    )
    probs_actual = actual_df.orderBy("test_id", "phrase_pos").select("probas").collect()
    probs_expected = (
        expected_df.orderBy("test_id", "phrase_pos").select("probas").collect()
    )
    # Check probabilities are within floating point error
    assert_are_close(probs_actual, probs_expected)
    # Check predicted labels
    assert is_equal_df(
        expected_df.select("test_id", "phrase_pos", "pred_modality"),
        actual_df.select("test_id", "phrase_pos", "pred_modality"),
        sort_columns=["test_id", "phrase_pos"],
    )


def test_avg_probability():
    res_unaggregated_df = infer.infer(traces_df, model_file=MODEL_PATH, aggregate=False)
    actual_df = infer.avg_probability(res_unaggregated_df, "id", "probas", 3)
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_avg_probability.json"))
    probs_actual = actual_df.orderBy("id").select("sentence_probas").collect()
    probs_expected = expected_df.orderBy("id").select("sentence_probas").collect()
    assert_are_close(probs_actual, probs_expected)
    assert is_equal_df(
        expected_df.select("id", "sentence_pred_label"),
        actual_df.select("id", "sentence_pred_label"),
        sort_columns=["id"],
    )
