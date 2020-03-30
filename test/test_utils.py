"""Test module for trace_classifier/utils.py"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')

import unittest
from os import path
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from trace_classifier import utils
from .utils import FIXTURES_PATH, is_equal_df

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


# def test_pad():
#     df = spark.createDataFrame([
#         ([[1, 2, 3], [3, 1, 2]],),
#         ([[1, 23, 3], [1, 221, 3]],),
#         ([[122, 2, 3], [122, 2, 3]],)
#     ], ["x"])
#     logging.info(df.withColumn(
#         "x_padded",
#         utils.pad(col("x"), lit(2))
#     ).collect())
#     assert False


def test_create_label_and_reverse():
    classes = ["Not Driving", "Driving", "Noise"]
    pred_unaggregated_df = spark.read.json(path.join(FIXTURES_PATH, "res_infer_unaggregated.json"))
    actual_df_forward = utils.create_label(pred_unaggregated_df, "pred_modality", "label", classes)
    expected_df_forward = spark.read.json(path.join(FIXTURES_PATH, "res_create_labels.json"))
    assert is_equal_df(expected_df_forward, actual_df_forward, ["id", "phrase_pos"])

    actual_df_backwards = utils.reverse_create_label(actual_df_forward, "label", "class", classes)
    reconstructed_classes = actual_df_backwards.select("class").rdd.flatMap(lambda x: x).collect()
    original_classes = pred_unaggregated_df.select("pred_modality").rdd.flatMap(lambda x: x).collect()
    assert all([ u == v for (u, v) in zip(original_classes, reconstructed_classes)])


def test_explode_array():
    df = spark.createDataFrame([
        ([1, 2, 3], "foo"),
        ([4, 5, 6, 7, 8], "bar")
    ], ["id", "baz"])
    expected = [{"baz":"foo","key":0,"value":1}, {"baz":"foo","key":1,"value":2}, {"baz":"foo","key":2,"value":3}, {"baz":"bar","key":0,"value":4}, {"baz":"bar","key":1,"value":5}, {"baz":"bar","key":2,"value":6}, {"baz":"bar","key":3,"value":7}, {"baz":"bar","key":4,"value":8}]
    actual = [ r.asDict() for r in utils.explode_array(df, "id", ["key", "value"]).collect() ]
    assert expected == actual
