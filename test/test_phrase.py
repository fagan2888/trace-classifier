"""Test module for trace_classifier/phrase.py"""
import logging
from os import path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ [%(levelname)s][%(name)s] %(message)s"
)

from pyspark.sql import SparkSession

from trace_classifier import word_vec
from trace_classifier import phrase
from .utils import FIXTURES_PATH, assert_are_close


spark = (
    SparkSession.builder.config(
        "spark.jars.packages", "databricks:tensorframes:0.5.0-s_2.11"
    )
    .enableHiveSupport()
    .appName("Generate routes test")
    .getOrCreate()
)


def test_create_phrases():
    word_vec_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_word_vecs.json"))
    actual_df = phrase.create_phrases(word_vec_df, "word_vec", "test_id", "word_pos")
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "res_create_phrases.json"))
    expected_phrase = (
        expected_df.orderBy("test_id", "phrase_pos").select("phrase").collect()
    )
    actual_phrase = (
        actual_df.orderBy("test_id", "phrase_pos").select("phrase").collect()
    )
    assert_are_close(expected_phrase, actual_phrase)

