"""Test module for trace_classifier/utils.py"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')

import unittest
from os import path
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from trace_classifier import utils
from .utils import FIXTURES_PATH

spark = SparkSession.builder \
    .config('spark.jars.packages', 'databricks:tensorframes:0.5.0-s_2.11') \
    .enableHiveSupport() \
    .appName("Generate routes test") \
    .getOrCreate()


def test_round_columns():
    df = spark.createDataFrame(
        [ (i + i**2/1000, i**2 + i**3/1000) for i in range(10) ],
        ["x", "y"]
    )
    res_df = utils.round_columns(df, ["x", "y"], 2).orderBy("x")
    xs = res_df.select("x").rdd.flatMap(lambda x: x).collect()
    ys = res_df.select("y").rdd.flatMap(lambda y: y).collect()

    assert all([round(x, 2) == x for x in xs])
    assert all([round(y, 2) == y for y in ys])

    res_df = utils.round_columns(df, ["x", "y"], 1).orderBy("x")
    xs = res_df.select("x").rdd.flatMap(lambda x: x).collect()
    ys = res_df.select("y").rdd.flatMap(lambda y: y).collect()

    assert all([round(x, 1) == x for x in xs])
    assert all([round(y, 1) == y for y in ys])


def test_clip():
    df = spark.createDataFrame(
        [ (i + i**2/1000, i**2 + i**3/1000) for i in range(10) ],
        ["x", "y"]
    )
    res_df = utils.clip(df, ["x"], [5.0, 10.0])
    assert res_df.where((col("x") < 5) | (col("x") > 10)).count() == 0
    xs = res_df.select("x").rdd.flatMap(lambda x: x).collect()
    assert all([x == (5.0 if x < 5.0 else (x if x <= 10.0 else 10.0)) for x in xs])

    res_df = utils.clip(df.withColumn("x", col("x") - 5), ["x", "y"], [-1, 1])
    xs = res_df.select("x").rdd.flatMap(lambda x: x).collect()
    ys = res_df.select("y").rdd.flatMap(lambda y: y).collect()
    assert all([x == (-1 if x < -1 else (x if x <= 1 else 1)) for x in xs])
    assert all([y == (-1 if y < -1 else (y if y <= 1 else 1)) for y in ys])


def test_argmax():
    df = spark.createDataFrame([
        ([1, 2, 3],),
        ([3, 1, 2],),
        ([1, 23, 3],),
        ([1, 221, 3],),
        ([122, 2, 3],)
    ], ["x"])
    res = [r.idx for r in df.select(utils.argmax(col("x")).alias("idx")).collect()]
    assert res == [2, 0, 1, 1, 0]


def test_pad():
    df = spark.createDataFrame([
        ([1, 2, 3],),
        ([3, 1, 2],),
        ([1, 23, 3],),
        ([1, 221, 3],),
        ([122, 2, 3],)
    ], ["x"])
    traces_df = spark.read.json(path.join(FIXTURES_PATH, "traces.json"))
    logging.info(traces_df.select(utils.pad(col("coordinates"), lit(2))).collect())


#
#
# def test_create_label():
#
#
# def test_reverse_create_label():
#
#
# def test_add_id():
#
