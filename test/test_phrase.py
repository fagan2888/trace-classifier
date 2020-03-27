"""Test module for trace_classifier/phrase.py"""
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

from trace_classifier import phrase
from .utils import FIXTURES_PATH, MODEL_PATH, TRACES_SCHEMA, is_equal_df, assert_are_close


spark = SparkSession.builder \
    .config('spark.jars.packages', 'databricks:tensorframes:0.5.0-s_2.11') \
    .enableHiveSupport() \
    .appName("Generate routes test") \
    .getOrCreate()


def test_create_phrases():
    word_vec_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_word_vecs.json"))
    actual_df = phrase.create_phrases(word_vec_df, "word_vec", "test_id", "word_pos")
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_phrases.json"))
    expected_phrase = expected_df.orderBy("test_id", "phrase_pos").select("phrase").collect()
    actual_phrase = actual_df.orderBy("test_id", "phrase_pos").select("phrase").collect()
    assert_are_close(expected_phrase, actual_phrase)
    # assert is_equal_df(expected_df, actual_df, sort_columns=["test_id", "phrase_pos"])
