"""Test module for trace_classifier/word_vec.py"""
import logging
from os import path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ [%(levelname)s][%(name)s] %(message)s"
)

from pyspark.sql import SparkSession

from trace_classifier import word_vec

from .utils import FIXTURES_PATH, TRACES_SCHEMA, is_equal_df

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


def test_create_words():
    # Traces w/ >= 3 coordinates
    actual_df = word_vec.create_words(traces_df, "coordinates", [3, 1, 2])
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_words.json"))
    assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "word_pos"])

    # Traces w/ < 3 coordinates are dropped
    actual_df = word_vec.create_words(short_traces_df, "coordinates", [3, 1, 2])
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_words_short.json"))
    assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "word_pos"])


def test_create_word_vecs():
    words_df = word_vec.create_words(traces_df, "coordinates", [3, 1, 2])

    desired_ops = [
        [
            (
                "dx",
                0,
                1,
            ),  # is like x(w[1]) - x(w[0]) where w[i] is the ith coordinate of the word w
            ("dy", 0, 1),
            ("d", 0, 1),
            ("t", 0, 1),
            ("s", 0, 1),
        ]
    ]
    actual_df, offsets, scales = word_vec.create_word_vecs(
        words_df, "word", desired_ops
    )
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_word_vecs.json"))
    assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "word_pos"])

    desired_ops = [
        [
            ("dx", 1, 2),
            ("dy", 0, 1),
            ("d", 0, 2),
            ("t", 0, 2),
            ("s", 2, 1),
        ]  # should be allowed
    ]
    actual_df, offsets, scales = word_vec.create_word_vecs(
        words_df, "word", desired_ops
    )
    expected_df = spark.read.json(
        path.join(FIXTURES_PATH, "res_create_word_vecs_1.json")
    )
    assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "word_pos"])
