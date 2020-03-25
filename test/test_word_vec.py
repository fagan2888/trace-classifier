"""Test module for trace_classifier/word_vec.py"""
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

from trace_classifier import word_vec

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

def test_create_words():
    actual_df = word_vec.create_words(traces_df, "coordinates", [3, 1, 2])
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_words.json"))
    assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "word_pos"])


def test_create_word_vecs():
    words_df = word_vec.create_words(traces_df, "coordinates", [3, 1, 2])
    desired_ops = [
        [("dx", 0, 1),
        ("dy", 0, 1),
        ("d", 0, 1),
        ("t", 0, 1),
        ("s", 0, 1)]
    ]
    actual_df, offsets, scales = word_vec.create_word_vecs(words_df, "word", desired_ops)
    logging.info(actual_df.toJSON().collect())
