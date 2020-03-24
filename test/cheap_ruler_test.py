"""Test module for trace_classifier/cheap_ruler.py"""
from os import path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')
import os
from unittest import mock

import findspark
findspark.init()

from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from .utils import is_equal_df

spark = SparkSession.builder \
    .config('spark.jars.packages', 'databricks:tensorframes:0.5.0-s_2.11') \
    .enableHiveSupport() \
    .appName("Generate routes test") \
    .getOrCreate()

from trace_classifier import cheap_ruler

COORDINATES_TYPE = T.ArrayType(T.DoubleType())
LINESTRING_TYPE = T.ArrayType(COORDINATES_TYPE)
FIXTURES_PATH = path.join(path.dirname(path.realpath(__file__)), "fixtures")
TRACES_SCHEMA = T.StructType([
    T.StructField("test_id", T.StringType()),
    T.StructField("coordinates", LINESTRING_TYPE)
])
RES_SCHEMA = T.StructType([
    *TRACES_SCHEMA.fields,
    T.StructField("kx", T.DoubleType()),
    T.StructField("ky", T.DoubleType())
])

def test_cheap_ruler():
    traces_df = spark.read.json(
        path.join(FIXTURES_PATH, "traces.json"),
        schema=TRACES_SCHEMA
    )
    actual_df = cheap_ruler.cheap_ruler(traces_df)
    expected_df = spark.read.json(path.join(FIXTURES_PATH, "./res_cheap_ruler.json"))
    assert is_equal_df(expected_df, actual_df)
