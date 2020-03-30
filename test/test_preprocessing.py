"""Test module for trace_classifier/preprocessing.py"""
from os import path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')
import os
from unittest import mock

import numpy as np

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from trace_classifier import preprocessing as pp
from trace_classifier import load
from .utils import FIXTURES_PATH, MODEL_PATH, TRACES_SCHEMA, is_equal_df, assert_are_close


spark = SparkSession.builder \
    .config('spark.jars.packages', 'databricks:tensorframes:0.5.0-s_2.11') \
    .enableHiveSupport() \
    .appName("Generate routes test") \
    .getOrCreate()

traces_df = spark.read.json(
    path.join(FIXTURES_PATH, "traces.json"),
    schema=TRACES_SCHEMA
)

def test_include_id_and_label():
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_include_id_and_label.json"))
    actual_df = pp.include_id_and_label(traces_df)
    assert is_equal_df(expected_df, actual_df)


def test_preprocessing_part2():
    metadata = load.load_model_metadata(MODEL_PATH)
    assert metadata is not None
    with_ids_and_labels_df = spark.read.json(path.join(FIXTURES_PATH, "res_include_id_and_label.json"))
    actual_df, actual_offsets, actual_scales  = pp.preprocessing_part2(with_ids_and_labels_df, metadata)
    # expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_preprocessing1.json"))
    # assert is_equal_df(expected_df, actual_df)
    logging.info(actual_df.toJSON().collect())
    logging.info(actual_offsets)
    logging.info(actual_scales)
    assert False
