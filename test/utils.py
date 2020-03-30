import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)sZ [%(levelname)s][%(name)s] %(message)s')
from os import path
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import numpy as np

FIXTURES_PATH = path.join(path.dirname(path.realpath(__file__)), "fixtures")
MODEL_PATH = path.join(FIXTURES_PATH, "sample_model_optimised_frozen.pb")
METADATA_PATH = path.join(FIXTURES_PATH, "sample_model_metadata.json")

COORDINATES_TYPE = T.ArrayType(T.DoubleType())
LINESTRING_TYPE = T.ArrayType(COORDINATES_TYPE)
TRACES_SCHEMA = T.StructType([
    T.StructField("test_id", T.StringType()),
    T.StructField("coordinates", LINESTRING_TYPE)
])

def is_equal_df(expected_df, actual_df, sort_columns=["test_id"]):
    """
    Test equality of two Spark DataFrames according to their collected contents
    Args:
        expected_df (pyspark.sql.DataFrame): first argument of equality relation
        actual_df (pyspark.sql.DataFrame): second argument of equality relation
        sort_column (String): the column by which to sort the two argument
    Returns
        bool
    """
    expected = expected_df.select(*sorted(expected_df.columns)) \
        .orderBy(*sort_columns).collect()
    actual = actual_df.select(*sorted(actual_df.columns)) \
        .orderBy(*sort_columns).collect()
    # e = expected[0]
    # a = actual[0]
    # logging.info(e)
    # logging.info(a)
    return expected == actual


def assert_are_close(actual, expected):
    """
    Test two array of arrays for element-wise equality up to floating point error.
    Args:
        actual: <array<array<number>>
        expected: <array<array<number>>
    Returns
        bool
    """
    # nested comprehension flattens all comparisons
    assert all([ r for (X_e, X_a) in zip(actual, expected) for r in np.isclose(X_e, X_a).flatten() ])
