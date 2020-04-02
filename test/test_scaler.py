"""Test module for trace_classifier/scaler.py"""
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)sZ [%(levelname)s][%(name)s] %(message)s"
)

from pyspark.sql import SparkSession
from trace_classifier import scaler


spark = (
    SparkSession.builder.config(
        "spark.jars.packages", "databricks:tensorframes:0.5.0-s_2.11"
    )
    .enableHiveSupport()
    .appName("Generate routes test")
    .getOrCreate()
)

df = spark.createDataFrame([(i, i ** 2) for i in range(10)], ["x", "y"])


def test_compute_mean():
    assert scaler.compute_mean(df, ["x"]) == 4.5
    assert scaler.compute_mean(df, ["y"]) == 28.5
    assert scaler.compute_mean(df, ["x", "y"]) == 16.5


def test_compute_mad():
    assert scaler.compute_mad(df, ["x"]) == 2.5
    assert scaler.compute_mad(df, ["y"]) == 23.2
    assert scaler.compute_mad(df, ["x", "y"]) == 17.25
    assert scaler.compute_mad(df, ["x"], mean_val=4.5) == 2.5
    assert scaler.compute_mad(df, ["y"], mean_val=28.5) == 23.2
    assert scaler.compute_mad(df, ["x", "y"], mean_val=16.5) == 17.25
